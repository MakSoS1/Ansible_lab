package com.dancecoach.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.webkit.WebViewAssetLoader
import com.dancecoach.mobile.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var pendingWebPermissionRequest: PermissionRequest? = null
    private lateinit var assetLoader: WebViewAssetLoader

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                pendingWebPermissionRequest?.grant(arrayOf(PermissionRequest.RESOURCE_VIDEO_CAPTURE))
            } else {
                setStatus(getString(R.string.status_need_camera))
                pendingWebPermissionRequest?.deny()
            }
            pendingWebPermissionRequest = null
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupWebView(binding.webView)
        setupInputControls()

        val fromIntent = intent?.dataString.orEmpty()
        if (fromIntent.isNotBlank()) {
            binding.editConnectUrl.setText(fromIntent)
        } else {
            val saved = getSharedPreferences("dance_coach_mobile", MODE_PRIVATE)
                .getString("connect_url", "")
                .orEmpty()
            if (saved.isNotBlank()) {
                binding.editConnectUrl.setText(saved)
            }
        }
    }

    private fun setupInputControls() {
        binding.btnConnect.setOnClickListener {
            loadTrackerPage()
        }

        binding.editConnectUrl.setOnEditorActionListener { _, actionId, event ->
            val done = actionId == EditorInfo.IME_ACTION_DONE
            val enter = event?.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN
            if (done || enter) {
                loadTrackerPage()
                true
            } else {
                false
            }
        }
    }

    private fun setupWebView(webView: WebView) {
        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.cacheMode = WebSettings.LOAD_DEFAULT
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.allowFileAccessFromFileURLs = true
        settings.allowUniversalAccessFromFileURLs = true
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                runOnUiThread {
                    val wantsCamera = request.resources.contains(PermissionRequest.RESOURCE_VIDEO_CAPTURE)
                    if (!wantsCamera) {
                        request.grant(request.resources)
                        return@runOnUiThread
                    }

                    if (ContextCompat.checkSelfPermission(
                            this@MainActivity,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        request.grant(request.resources)
                    } else {
                        pendingWebPermissionRequest = request
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            }
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView?,
                request: WebResourceRequest
            ): WebResourceResponse? {
                return assetLoader.shouldInterceptRequest(request.url)
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    setStatus("WebView error: ${error?.description ?: "unknown"}")
                }
            }
        }
    }

    private fun loadTrackerPage() {
        val raw = binding.editConnectUrl.text?.toString()?.trim().orEmpty()
        if (raw.isBlank()) {
            setStatus(getString(R.string.status_invalid_url))
            return
        }

        val parsed = runCatching { Uri.parse(raw) }.getOrNull()
        if (parsed == null) {
            setStatus(getString(R.string.status_invalid_url))
            return
        }

        val sessionId = parsed.getQueryParameter("session").orEmpty()
        if (sessionId.isBlank()) {
            setStatus(getString(R.string.status_missing_session))
            return
        }

        val host = parsed.authority.orEmpty()
        val scheme = parsed.scheme.orEmpty().ifBlank { "http" }
        if (host.isBlank() || (scheme != "http" && scheme != "https")) {
            setStatus(getString(R.string.status_invalid_url))
            return
        }
        val serverOrigin = "$scheme://$host"

        getSharedPreferences("dance_coach_mobile", MODE_PRIVATE)
            .edit()
            .putString("connect_url", raw)
            .apply()

        val localUrl = "https://appassets.androidplatform.net/assets/mobile/index.html?session=${Uri.encode(sessionId)}&server=${Uri.encode(serverOrigin)}"
        setStatus(getString(R.string.status_loading, sessionId, host))
        binding.webView.loadUrl(localUrl)
    }

    private fun setStatus(text: String) {
        binding.textStatus.text = text
    }
}
