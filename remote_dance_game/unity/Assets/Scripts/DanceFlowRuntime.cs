using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.UI;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine.Video;

namespace DanceFlow.UnityClient
{
    [Serializable] public class DanceTheme { public string name; public int[] primary; public int[] secondary; public int[] accent; public int[] deep; public string motif; }
    [Serializable] public class DanceListItem { public string dance_id; public string title; public int duration_ms; public string difficulty; public string created_at; public bool has_video; public bool has_poster; public string preview_mode; public DanceTheme theme; }
    [Serializable] public class DanceDetail { public string dance_id; public string title; public int version; public int duration_ms; public string skeleton_format; public string preview_mode; public string difficulty; public bool mirror_mode; public string created_at; public int num_frames; public int num_events; public bool has_poster; public DanceTheme theme; }
    [Serializable] public class SessionCreateRequest { public string dance_id; }
    [Serializable] public class GameSession { public string session_id; public string dance_id; public string connect_url; public string qr_data; }
    [Serializable] public class SessionStatus { public string status; public bool phone_connected; public bool calibrated; }
    [Serializable] public class Landmark { public float x; public float y; public float z; public float v; }
    [Serializable] public class PoseTimelineFrame { public int t_ms; public Landmark[] landmarks; }
    [Serializable] public class PoseTimeline { public string dance_id; public int fps; public int duration_ms; public string space; public PoseTimelineFrame[] frames; }
    [Serializable] public class GameResults { public int total_score; public int max_combo; public float accuracy_arms; public float accuracy_legs; public float accuracy_torso; }
    [Serializable] public class SocketEnvelope { public string type; public string grade; public int timestamp_ms; public int score; public int total_score; public int combo; public float similarity; public string hold_state; public float timing_offset_ms; public bool tracking_lost; public GameResults results; }
    [Serializable] public class MediaClockMessage { public string action = "media_clock"; public int media_time_ms; }
    [Serializable] public class GameActionMessage { public string action; public int media_time_ms; }

    public static class AppConfig
    {
        public static string ApiBase
        {
            get
            {
                string value = Environment.GetEnvironmentVariable("DANCE_API_BASE");
                if (!string.IsNullOrWhiteSpace(value)) return value.TrimEnd('/');
                return PlayerPrefs.GetString("dance_api_base", "http://127.0.0.1:8000").TrimEnd('/');
            }
        }
        public const int ReferenceWidth = 3840;
        public const int ReferenceHeight = 2160;
        public const int TargetFps = 60;
    }

    public sealed class DanceApiClient
    {
        [Serializable] private class ArrayWrapper<T> { public T[] items; }
        public string BaseUrl { get; private set; }
        public DanceApiClient(string baseUrl) { BaseUrl = baseUrl.TrimEnd('/'); }

        private static async Task WaitAsync(UnityWebRequest request)
        {
            UnityWebRequestAsyncOperation op = request.SendWebRequest();
            while (!op.isDone) await Task.Yield();
            if (request.result != UnityWebRequest.Result.Success)
                throw new Exception("HTTP " + request.responseCode + ": " + request.error + " " + request.downloadHandler.text);
        }

        public async Task<T> GetAsync<T>(string path)
        {
            using (UnityWebRequest req = UnityWebRequest.Get(BaseUrl + path))
            {
                await WaitAsync(req);
                return JsonUtility.FromJson<T>(req.downloadHandler.text);
            }
        }

        public async Task<T[]> GetArrayAsync<T>(string path)
        {
            using (UnityWebRequest req = UnityWebRequest.Get(BaseUrl + path))
            {
                await WaitAsync(req);
                string wrapped = "{\"items\":" + req.downloadHandler.text + "}";
                ArrayWrapper<T> parsed = JsonUtility.FromJson<ArrayWrapper<T>>(wrapped);
                return parsed != null && parsed.items != null ? parsed.items : Array.Empty<T>();
            }
        }

        public async Task<TResponse> PostAsync<TRequest, TResponse>(string path, TRequest body)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));
            using (UnityWebRequest req = new UnityWebRequest(BaseUrl + path, UnityWebRequest.kHttpVerbPOST))
            {
                req.uploadHandler = new UploadHandlerRaw(bytes);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                await WaitAsync(req);
                return JsonUtility.FromJson<TResponse>(req.downloadHandler.text);
            }
        }

        public async Task<Texture2D> LoadTextureAsync(string absoluteUrl)
        {
            using (UnityWebRequest req = UnityWebRequestTexture.GetTexture(absoluteUrl, true))
            {
                await WaitAsync(req);
                return DownloadHandlerTexture.GetContent(req);
            }
        }

        public string VideoUrl(string danceId) { return BaseUrl + "/video/" + danceId; }
        public string PosterUrl(string danceId) { return BaseUrl + "/api/dances/" + danceId + "/poster"; }
        public string PoseTimelineUrl(string danceId, int fps) { return "/api/dances/" + danceId + "/pose-timeline?fps=" + fps; }
        public string GameSocketUrl(string sessionId)
        {
            Uri http = new Uri(BaseUrl);
            string scheme = http.Scheme == "https" ? "wss" : "ws";
            return scheme + "://" + http.Authority + "/ws/game/" + sessionId;
        }
    }

    public sealed class GameSocketClient : IDisposable
    {
        private ClientWebSocket socket;
        private CancellationTokenSource cancellation;
        private readonly ConcurrentQueue<string> inbound = new ConcurrentQueue<string>();
        public bool IsConnected { get { return socket != null && socket.State == WebSocketState.Open; } }

        public async Task ConnectAsync(string url)
        {
            Dispose();
            socket = new ClientWebSocket();
            cancellation = new CancellationTokenSource();
            await socket.ConnectAsync(new Uri(url), cancellation.Token);
            _ = ReceiveLoop();
        }

        private async Task ReceiveLoop()
        {
            byte[] buffer = new byte[64 * 1024];
            while (socket != null && socket.State == WebSocketState.Open && !cancellation.IsCancellationRequested)
            {
                StringBuilder builder = new StringBuilder();
                WebSocketReceiveResult result;
                do
                {
                    result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellation.Token);
                    if (result.MessageType == WebSocketMessageType.Close) return;
                    builder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                }
                while (!result.EndOfMessage);
                inbound.Enqueue(builder.ToString());
            }
        }

        public void Drain(Action<SocketEnvelope> handler)
        {
            string json;
            while (inbound.TryDequeue(out json))
            {
                try
                {
                    SocketEnvelope message = JsonUtility.FromJson<SocketEnvelope>(json);
                    if (message != null) handler(message);
                }
                catch (Exception e) { Debug.LogWarning("Socket parse error: " + e.Message); }
            }
        }

        public async Task SendAsync(object payload)
        {
            if (!IsConnected) return;
            byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload));
            await socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cancellation.Token);
        }

        public void Dispose()
        {
            try { if (cancellation != null) cancellation.Cancel(); } catch { }
            try { if (socket != null) socket.Dispose(); } catch { }
            cancellation = null; socket = null;
            while (inbound.TryDequeue(out _)) { }
        }
    }

    public sealed class TvInputRouter : MonoBehaviour
    {
        public event Action Left; public event Action Right; public event Action Up; public event Action Down; public event Action Submit; public event Action Cancel;
        private InputAction left, right, up, down, submit, cancel;

        private void Awake()
        {
            left = NewAction("left", "<Keyboard>/leftArrow", "<Keyboard>/a", "<Gamepad>/dpad/left", "<Gamepad>/leftStick/left");
            right = NewAction("right", "<Keyboard>/rightArrow", "<Keyboard>/d", "<Gamepad>/dpad/right", "<Gamepad>/leftStick/right");
            up = NewAction("up", "<Keyboard>/upArrow", "<Keyboard>/w", "<Gamepad>/dpad/up", "<Gamepad>/leftStick/up");
            down = NewAction("down", "<Keyboard>/downArrow", "<Keyboard>/s", "<Gamepad>/dpad/down", "<Gamepad>/leftStick/down");
            submit = NewAction("submit", "<Keyboard>/enter", "<Keyboard>/space", "<Gamepad>/buttonSouth");
            cancel = NewAction("cancel", "<Keyboard>/escape", "<Keyboard>/backspace", "<Gamepad>/buttonEast");
        }

        private InputAction NewAction(string name, params string[] bindings)
        {
            InputAction action = new InputAction(name, InputActionType.Button);
            foreach (string binding in bindings) action.AddBinding(binding);
            return action;
        }

        private void OnEnable() { left.Enable(); right.Enable(); up.Enable(); down.Enable(); submit.Enable(); cancel.Enable(); }
        private void OnDisable() { left.Disable(); right.Disable(); up.Disable(); down.Disable(); submit.Disable(); cancel.Disable(); }
        private void Update()
        {
            if (left.WasPressedThisFrame()) Left?.Invoke();
            if (right.WasPressedThisFrame()) Right?.Invoke();
            if (up.WasPressedThisFrame()) Up?.Invoke();
            if (down.WasPressedThisFrame()) Down?.Invoke();
            if (submit.WasPressedThisFrame()) Submit?.Invoke();
            if (cancel.WasPressedThisFrame()) Cancel?.Invoke();
        }
    }

    public static class RuntimeUi
    {
        public static readonly Color Deep = new Color32(8, 5, 27, 255);
        public static readonly Color Glass = new Color32(19, 11, 52, 218);
        public static readonly Color Pink = new Color32(255, 42, 204, 255);
        public static readonly Color Cyan = new Color32(54, 228, 255, 255);
        public static readonly Color Gold = new Color32(255, 218, 83, 255);
        public static readonly Color White = new Color32(249, 249, 255, 255);
        private static Font cachedFont;
        public static Font Font { get { if (cachedFont == null) cachedFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); return cachedFont; } }

        public static Canvas CreateCanvas(Transform parent, string name, int order)
        {
            GameObject go = new GameObject(name, typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            go.transform.SetParent(parent, false);
            Canvas canvas = go.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceOverlay; canvas.sortingOrder = order;
            CanvasScaler scaler = go.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(AppConfig.ReferenceWidth, AppConfig.ReferenceHeight);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight; scaler.matchWidthOrHeight = 0.5f;
            return canvas;
        }

        public static RectTransform Rect(Transform parent, string name, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
        {
            GameObject go = new GameObject(name, typeof(RectTransform)); go.transform.SetParent(parent, false);
            RectTransform r = go.GetComponent<RectTransform>(); r.anchorMin = anchorMin; r.anchorMax = anchorMax; r.offsetMin = offsetMin; r.offsetMax = offsetMax; return r;
        }

        public static RawImage Panel(Transform parent, string name, Color color, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
        {
            RectTransform r = Rect(parent, name, anchorMin, anchorMax, offsetMin, offsetMax);
            RawImage img = r.gameObject.AddComponent<RawImage>(); img.color = color; img.texture = Texture2D.whiteTexture; return img;
        }

        public static Text Label(Transform parent, string name, string text, int fontSize, TextAnchor alignment, Color color, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
        {
            RectTransform r = Rect(parent, name, anchorMin, anchorMax, offsetMin, offsetMax);
            Text label = r.gameObject.AddComponent<Text>(); label.font = Font; label.text = text; label.fontSize = fontSize; label.fontStyle = FontStyle.Bold;
            label.alignment = alignment; label.color = color; label.horizontalOverflow = HorizontalWrapMode.Overflow; label.verticalOverflow = VerticalWrapMode.Overflow; return label;
        }

        public static Button Button(Transform parent, string name, string text, Color color, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
        {
            RawImage bg = Panel(parent, name, color, anchorMin, anchorMax, offsetMin, offsetMax);
            Button button = bg.gameObject.AddComponent<Button>(); button.targetGraphic = bg;
            Label(bg.transform, "Label", text, 42, TextAnchor.MiddleCenter, White, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero); return button;
        }

        public static void EnsureEventSystem()
        {
            if (UnityEngine.Object.FindFirstObjectByType<EventSystem>() != null) return;
            GameObject go = new GameObject("EventSystem", typeof(EventSystem), typeof(InputSystemUIInputModule)); UnityEngine.Object.DontDestroyOnLoad(go);
        }

        public static async Task SetTextureAsync(RawImage target, DanceApiClient api, string url)
        {
            try { Texture2D texture = await api.LoadTextureAsync(url); if (target != null) target.texture = texture; }
            catch (Exception e) { Debug.LogWarning("Texture load failed: " + e.Message); }
        }

        public static Texture2D DecodeDataUrl(string dataUrl)
        {
            if (string.IsNullOrWhiteSpace(dataUrl)) return null;
            int comma = dataUrl.IndexOf(','); string encoded = comma >= 0 ? dataUrl.Substring(comma + 1) : dataUrl;
            byte[] bytes = Convert.FromBase64String(encoded); Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false); texture.LoadImage(bytes); return texture;
        }
    }

    public sealed class DanceFlowApp : MonoBehaviour
    {
        public static DanceFlowApp Instance { get; private set; }
        public DanceApiClient Api { get; private set; }
        public TvInputRouter Input { get; private set; }
        public DanceListItem SelectedDance { get; private set; }
        public GameSession CurrentSession { get; private set; }
        private GameObject screen;

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this; DontDestroyOnLoad(gameObject);
            Api = new DanceApiClient(AppConfig.ApiBase); Input = gameObject.AddComponent<TvInputRouter>();
            QualitySettings.vSyncCount = 1; Application.targetFrameRate = AppConfig.TargetFps; Screen.fullScreenMode = FullScreenMode.FullScreenWindow;
            RuntimeUi.EnsureEventSystem();
        }
        private void Start() { ShowLibrary(); }

        private T NewScreen<T>(string name) where T : MonoBehaviour
        {
            if (screen != null) Destroy(screen);
            screen = new GameObject(name); screen.transform.SetParent(transform, false); return screen.AddComponent<T>();
        }
        public void ShowLibrary() { NewScreen<LibraryScreen>("LibraryScreen").Initialize(this); }
        public void ShowConnect(DanceListItem dance) { SelectedDance = dance; NewScreen<ConnectScreen>("ConnectScreen").Initialize(this, dance); }
        public void ShowGameplay(GameSession session) { CurrentSession = session; NewScreen<GameplayScreen>("GameplayScreen").Initialize(this, SelectedDance, session); }
    }

    public sealed class LibraryScreen : MonoBehaviour
    {
        private DanceFlowApp app; private Canvas canvas; private RawImage backdrop; private RawImage previewSurface; private VideoPlayer previewPlayer; private RenderTexture previewTexture;
        private readonly List<Card> cards = new List<Card>(); private DanceListItem[] dances = Array.Empty<DanceListItem>(); private int selected; private int previewToken;
        private Text selectedTitle; private Text selectedMeta;
        private sealed class Card { public GameObject root; public RectTransform rect; public RawImage art; public Text title; }

        public void Initialize(DanceFlowApp value) { app = value; BuildUi(); HookInput(); _ = LoadAsync(); }

        private void BuildUi()
        {
            canvas = RuntimeUi.CreateCanvas(transform, "TV Library", 0);
            backdrop = RuntimeUi.Panel(canvas.transform, "Backdrop", RuntimeUi.Deep, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            previewSurface = RuntimeUi.Panel(canvas.transform, "Video Preview", Color.black, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero); previewSurface.color = new Color(1f, 1f, 1f, 0.78f);
            RuntimeUi.Panel(canvas.transform, "Vignette", new Color(0.025f, 0.015f, 0.08f, 0.44f), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            RuntimeUi.Label(canvas.transform, "Brand", "DANCEFLOW", 118, TextAnchor.MiddleLeft, RuntimeUi.White, new Vector2(0, 1), new Vector2(0, 1), new Vector2(72, -170), new Vector2(850, -35));
            RuntimeUi.Label(canvas.transform, "Tabs", "SONGS     PLAYLISTS     CHALLENGES     PARTY", 38, TextAnchor.MiddleCenter, RuntimeUi.White, new Vector2(0.30f, 1), new Vector2(0.70f, 1), new Vector2(0, -125), new Vector2(0, -50));
            RuntimeUi.Label(canvas.transform, "Profile", "PLAYER   Lv. 12       ★ 1,260", 36, TextAnchor.MiddleRight, RuntimeUi.White, new Vector2(0.72f, 1), new Vector2(1, 1), new Vector2(0, -130), new Vector2(-68, -45));
            selectedTitle = RuntimeUi.Label(canvas.transform, "SelectedTitle", "", 88, TextAnchor.MiddleCenter, RuntimeUi.White, new Vector2(0.20f, 0), new Vector2(0.80f, 0), new Vector2(0, 285), new Vector2(0, 410));
            selectedMeta = RuntimeUi.Label(canvas.transform, "SelectedMeta", "", 30, TextAnchor.MiddleCenter, new Color(0.84f, 0.84f, 0.94f), new Vector2(0.20f, 0), new Vector2(0.80f, 0), new Vector2(0, 240), new Vector2(0, 300));
            RuntimeUi.Button(canvas.transform, "PlayButton", "▶  PLAY", RuntimeUi.Pink, new Vector2(0.39f, 0), new Vector2(0.61f, 0), new Vector2(0, 95), new Vector2(0, 205));
            RuntimeUi.Label(canvas.transform, "Hints", "◀  ▶   SELECT        A / ENTER  PLAY        B / ESC  BACK", 24, TextAnchor.MiddleCenter, new Color(1, 1, 1, 0.75f), new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 26), new Vector2(0, 72));
            int rw = Screen.width >= 3000 ? 3840 : 1920; int rh = Screen.width >= 3000 ? 2160 : 1080;
            previewTexture = new RenderTexture(rw, rh, 0, RenderTextureFormat.ARGB32);
            previewPlayer = gameObject.AddComponent<VideoPlayer>(); previewPlayer.playOnAwake = false; previewPlayer.isLooping = true;
            previewPlayer.renderMode = VideoRenderMode.RenderTexture; previewPlayer.targetTexture = previewTexture; previewPlayer.audioOutputMode = VideoAudioOutputMode.Direct; previewSurface.texture = previewTexture;
        }

        private async Task LoadAsync()
        {
            try
            {
                dances = await app.Api.GetArrayAsync<DanceListItem>("/api/dances");
                if (dances.Length == 0) { selectedTitle.text = "NO DANCES YET"; selectedMeta.text = "Import a dance with the existing builder, then return here."; return; }
                CreateCards(); Select(0);
            }
            catch (Exception e) { selectedTitle.text = "BACKEND OFFLINE"; selectedMeta.text = e.Message; }
        }

        private void CreateCards()
        {
            foreach (Card card in cards) Destroy(card.root); cards.Clear();
            for (int i = 0; i < dances.Length; i++)
            {
                GameObject root = new GameObject("Card_" + i, typeof(RectTransform), typeof(RawImage)); root.transform.SetParent(canvas.transform, false);
                RectTransform rect = root.GetComponent<RectTransform>(); RawImage panel = root.GetComponent<RawImage>(); panel.texture = Texture2D.whiteTexture; panel.color = new Color(0.06f, 0.03f, 0.14f, 0.98f);
                RawImage art = RuntimeUi.Panel(root.transform, "Art", Color.white, Vector2.zero, Vector2.one, new Vector2(14, 105), new Vector2(-14, -14));
                Text title = RuntimeUi.Label(root.transform, "Title", dances[i].title, 36, TextAnchor.MiddleCenter, RuntimeUi.White, new Vector2(0, 0), new Vector2(1, 0), new Vector2(20, 20), new Vector2(-20, 100));
                Card card = new Card { root = root, rect = rect, art = art, title = title }; cards.Add(card);
                if (dances[i].has_poster) _ = RuntimeUi.SetTextureAsync(art, app.Api, app.Api.PosterUrl(dances[i].dance_id));
            }
        }

        private void Select(int index)
        {
            if (dances.Length == 0) return;
            selected = (index % dances.Length + dances.Length) % dances.Length;
            for (int i = 0; i < cards.Count; i++)
            {
                int delta = i - selected; Card card = cards[i]; bool active = delta == 0; float x = delta * 720f;
                card.rect.anchorMin = card.rect.anchorMax = new Vector2(0.5f, 0.52f); card.rect.pivot = new Vector2(0.5f, 0.5f);
                card.rect.anchoredPosition = new Vector2(x, active ? 30 : -10); card.rect.sizeDelta = active ? new Vector2(920, 810) : new Vector2(650, 570);
                card.root.SetActive(Mathf.Abs(delta) <= 2); card.root.GetComponent<RawImage>().color = active ? new Color(0.16f, 0.06f, 0.30f, 1f) : new Color(0.045f, 0.025f, 0.12f, 0.95f);
                card.title.fontSize = active ? 48 : 34;
            }
            DanceListItem dance = dances[selected]; selectedTitle.text = dance.title;
            int minutes = dance.duration_ms / 60000; int seconds = dance.duration_ms / 1000 % 60;
            selectedMeta.text = dance.difficulty.ToUpperInvariant() + "     " + minutes + ":" + seconds.ToString("00") + "     SOLO";
            _ = UpdateBackdropAsync(dance);
        }

        private async Task UpdateBackdropAsync(DanceListItem dance)
        {
            int token = ++previewToken;
            if (dance.has_poster)
            {
                try { Texture2D poster = await app.Api.LoadTextureAsync(app.Api.PosterUrl(dance.dance_id)); if (token == previewToken && backdrop != null) backdrop.texture = poster; } catch { }
            }
            await Task.Delay(850);
            if (token != previewToken || previewPlayer == null) return;
            previewPlayer.Stop(); previewPlayer.url = app.Api.VideoUrl(dance.dance_id); previewPlayer.SetDirectAudioVolume(0, 0.07f); previewPlayer.Play();
        }

        private void HookInput() { app.Input.Left += Prev; app.Input.Right += Next; app.Input.Submit += Play; }
        private void UnhookInput() { if (app == null || app.Input == null) return; app.Input.Left -= Prev; app.Input.Right -= Next; app.Input.Submit -= Play; }
        private void Prev() { Select(selected - 1); } private void Next() { Select(selected + 1); } private void Play() { if (dances.Length > 0) app.ShowConnect(dances[selected]); }
        private void OnDestroy() { UnhookInput(); if (previewPlayer != null) previewPlayer.Stop(); if (previewTexture != null) previewTexture.Release(); }
    }

    public sealed class ConnectScreen : MonoBehaviour
    {
        private DanceFlowApp app; private DanceListItem dance; private GameSession session; private Text status; private RawImage qr; private bool ready; private int generation;
        public void Initialize(DanceFlowApp value, DanceListItem selected) { app = value; dance = selected; BuildUi(); app.Input.Cancel += Back; app.Input.Submit += TryStart; _ = SetupAsync(++generation); }

        private void BuildUi()
        {
            Canvas canvas = RuntimeUi.CreateCanvas(transform, "Connect", 0);
            RawImage background = RuntimeUi.Panel(canvas.transform, "Background", RuntimeUi.Deep, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            if (dance.has_poster) _ = RuntimeUi.SetTextureAsync(background, app.Api, app.Api.PosterUrl(dance.dance_id));
            RuntimeUi.Panel(canvas.transform, "Shade", new Color(0.025f, 0.01f, 0.08f, 0.63f), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            RuntimeUi.Label(canvas.transform, "Brand", "DANCEFLOW", 100, TextAnchor.MiddleLeft, RuntimeUi.White, new Vector2(0, 1), new Vector2(0, 1), new Vector2(70, -155), new Vector2(780, -40));
            RuntimeUi.Label(canvas.transform, "Step", "ONE QUICK STEP", 32, TextAnchor.MiddleLeft, RuntimeUi.Cyan, new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(180, 220), new Vector2(1450, 300));
            RuntimeUi.Label(canvas.transform, "Title", "POINT YOUR PHONE\nAT THE ROOM", 94, TextAnchor.MiddleLeft, RuntimeUi.White, new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(180, -20), new Vector2(1750, 220));
            RuntimeUi.Label(canvas.transform, "Copy", "Keep your full body visible. The phone sends pose points only.", 34, TextAnchor.MiddleLeft, new Color(0.90f, 0.90f, 0.98f), new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(180, -150), new Vector2(1600, -55));
            RawImage qrPanel = RuntimeUi.Panel(canvas.transform, "QRPanel", new Color(1, 1, 1, 0.94f), new Vector2(0.71f, 0.5f), new Vector2(0.71f, 0.5f), new Vector2(-330, -330), new Vector2(330, 330));
            qr = RuntimeUi.Panel(qrPanel.transform, "QR", Color.white, Vector2.zero, Vector2.one, new Vector2(48, 48), new Vector2(-48, -48));
            status = RuntimeUi.Label(canvas.transform, "Status", "Creating session…", 38, TextAnchor.MiddleLeft, RuntimeUi.Gold, new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(180, -300), new Vector2(1500, -230));
            RuntimeUi.Label(canvas.transform, "Hint", "B / ESC  BACK", 24, TextAnchor.MiddleLeft, new Color(1,1,1,.72f), new Vector2(0,0), new Vector2(0,0), new Vector2(60,35), new Vector2(500,85));
        }

        private async Task SetupAsync(int token)
        {
            try
            {
                session = await app.Api.PostAsync<SessionCreateRequest, GameSession>("/api/session/create", new SessionCreateRequest { dance_id = dance.dance_id });
                if (token != generation) return;
                Texture2D qrTexture = RuntimeUi.DecodeDataUrl(session.qr_data); if (qrTexture != null && qr != null) qr.texture = qrTexture;
                status.text = "SCAN QR CODE — WAITING FOR PHONE";
                while (token == generation && !ready)
                {
                    SessionStatus state = await app.Api.GetAsync<SessionStatus>("/api/session/" + session.session_id + "/status");
                    if (state.phone_connected)
                    {
                        if (state.calibrated) { ready = true; status.text = "✓ PHONE READY     PRESS A / ENTER TO DANCE"; break; }
                        status.text = "✓ PHONE CONNECTED     CALIBRATING…";
                    }
                    await Task.Delay(550);
                }
            }
            catch (Exception e) { if (status != null) status.text = "SESSION ERROR: " + e.Message; }
        }

        private void TryStart() { if (ready && session != null) app.ShowGameplay(session); }
        private void Back() { app.ShowLibrary(); }
        private void OnDestroy() { generation++; if (app != null) { app.Input.Cancel -= Back; app.Input.Submit -= TryStart; } }
    }

    public sealed class GameplayScreen : MonoBehaviour
    {
        private DanceFlowApp app; private DanceListItem dance; private GameSession session; private GameSocketClient socket; private VideoPlayer video;
        private RenderTexture videoTexture; private RawImage videoSurface; private Text scoreText; private Text comboText; private Text gradeText; private Text progressText; private RawImage progressFill;
        private bool started; private bool paused; private float lastClockSend; private float gradeUntil;

        public void Initialize(DanceFlowApp value, DanceListItem selected, GameSession currentSession) { app = value; dance = selected; session = currentSession; BuildUi(); app.Input.Cancel += Exit; app.Input.Submit += TogglePause; _ = StartGameAsync(); }

        private void BuildUi()
        {
            Canvas canvas = RuntimeUi.CreateCanvas(transform, "Gameplay", 0);
            videoSurface = RuntimeUi.Panel(canvas.transform, "Video", Color.white, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            int rw = Screen.width >= 3000 ? 3840 : 1920; int rh = Screen.width >= 3000 ? 2160 : 1080;
            videoTexture = new RenderTexture(rw, rh, 0, RenderTextureFormat.ARGB32);
            video = gameObject.AddComponent<VideoPlayer>(); video.playOnAwake = false; video.url = app.Api.VideoUrl(dance.dance_id);
            video.renderMode = VideoRenderMode.RenderTexture; video.targetTexture = videoTexture; video.audioOutputMode = VideoAudioOutputMode.Direct; videoSurface.texture = videoTexture; video.loopPointReached += OnVideoFinished;
            RuntimeUi.Panel(canvas.transform, "Vignette", new Color(0.025f, 0.015f, 0.07f, 0.14f), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            RawImage song = RuntimeUi.Panel(canvas.transform, "SongPanel", new Color(0.03f,0.02f,0.11f,0.82f), new Vector2(0,1), new Vector2(0,1), new Vector2(55,-230), new Vector2(1320,-45));
            RawImage cover = RuntimeUi.Panel(song.transform, "Cover", Color.white, new Vector2(0,0), new Vector2(0,1), new Vector2(16,16), new Vector2(220,-16));
            if (dance.has_poster) _ = RuntimeUi.SetTextureAsync(cover, app.Api, app.Api.PosterUrl(dance.dance_id));
            RuntimeUi.Label(song.transform, "Title", dance.title, 58, TextAnchor.MiddleLeft, RuntimeUi.White, new Vector2(0,0.52f), new Vector2(1,1), new Vector2(255,0), new Vector2(-30,-5));
            progressText = RuntimeUi.Label(song.transform, "Time", "0:00", 30, TextAnchor.MiddleRight, new Color(.9f,.9f,.98f), new Vector2(.70f,0), new Vector2(1,0.48f), new Vector2(0,10), new Vector2(-30,0));
            RawImage bar = RuntimeUi.Panel(song.transform, "ProgressBar", new Color(1,1,1,.20f), new Vector2(0,0), new Vector2(1,0), new Vector2(255,35), new Vector2(-160,55));
            progressFill = RuntimeUi.Panel(bar.transform, "Fill", RuntimeUi.Pink, Vector2.zero, new Vector2(0,1), Vector2.zero, Vector2.zero);

            RawImage score = RuntimeUi.Panel(canvas.transform, "ScorePanel", new Color(0.03f,0.02f,0.11f,0.72f), new Vector2(1,1), new Vector2(1,1), new Vector2(-1190,-280), new Vector2(-55,-45));
            RuntimeUi.Label(score.transform, "Label", "♕  SCORE", 34, TextAnchor.UpperLeft, RuntimeUi.White, new Vector2(0,0.58f), new Vector2(0.5f,1), new Vector2(30,0), new Vector2(0,-18));
            scoreText = RuntimeUi.Label(score.transform, "Score", "0", 82, TextAnchor.MiddleLeft, RuntimeUi.White, new Vector2(0,0.15f), new Vector2(0.58f,0.74f), new Vector2(30,0), new Vector2(0,0));
            comboText = RuntimeUi.Label(score.transform, "Combo", "COMBO  0", 52, TextAnchor.MiddleCenter, RuntimeUi.White, new Vector2(0.58f,0.10f), new Vector2(1,0.62f), Vector2.zero, new Vector2(-20,0));
            RuntimeUi.Label(score.transform, "Stars", "★ ★ ★ ★ ★", 54, TextAnchor.UpperRight, RuntimeUi.Gold, new Vector2(0.50f,0.56f), new Vector2(1,1), Vector2.zero, new Vector2(-22,-12));

            for (int i = 0; i < 3; i++)
                RuntimeUi.Panel(canvas.transform, "NextMove" + i, new Color(0.12f,0.05f,0.24f,0.78f), new Vector2(1,0.5f), new Vector2(1,0.5f), new Vector2(-345, 170 - i * 295), new Vector2(-95, 420 - i * 295));
            RuntimeUi.Label(canvas.transform, "NextMovesLabel", "NEXT\nMOVES", 26, TextAnchor.MiddleCenter, new Color(.9f,.9f,1,.78f), new Vector2(1,.5f), new Vector2(1,.5f), new Vector2(-350,-720), new Vector2(-90,-620));

            gradeText = RuntimeUi.Label(canvas.transform, "Grade", "", 104, TextAnchor.MiddleCenter, RuntimeUi.Cyan, new Vector2(.69f,.47f), new Vector2(.88f,.66f), Vector2.zero, Vector2.zero);
            RawImage timeline = RuntimeUi.Panel(canvas.transform, "Timeline", new Color(0.04f,0.02f,0.12f,0.72f), new Vector2(.08f,0), new Vector2(.92f,0), new Vector2(0,45), new Vector2(0,155));
            RuntimeUi.Panel(timeline.transform, "Line", new Color(.78f,.93f,1,.72f), new Vector2(.05f,.5f), new Vector2(.95f,.5f), new Vector2(0,-3), new Vector2(0,3));
            CoachStage3D coach3D = gameObject.AddComponent<CoachStage3D>(); coach3D.Initialize(canvas, app.Api, dance.dance_id, video, videoSurface);
        }

        private async Task StartGameAsync()
        {
            try
            {
                socket = new GameSocketClient(); await socket.ConnectAsync(app.Api.GameSocketUrl(session.session_id)); await ShowCountdownAsync();
                video.SetDirectAudioVolume(0, 1.0f); video.Play(); await socket.SendAsync(new GameActionMessage { action = "start", media_time_ms = 0 }); started = true;
            }
            catch (Exception e) { gradeText.text = "CONNECTION ERROR\n" + e.Message; }
        }

        private async Task ShowCountdownAsync()
        {
            string[] values = { "3", "2", "1", "GO!" };
            foreach (string value in values) { gradeText.text = value; gradeText.color = value == "GO!" ? RuntimeUi.Gold : RuntimeUi.White; await Task.Delay(value == "GO!" ? 350 : 650); }
            gradeText.text = "";
        }

        private void Update()
        {
            if (socket != null) socket.Drain(HandleMessage);
            if (!started || video == null || !video.isPlaying) return;
            int currentMs = Mathf.Max(0, (int)(video.time * 1000.0)); int duration = Mathf.Max(dance.duration_ms, 1); float p = Mathf.Clamp01(currentMs / (float)duration);
            if (progressFill != null) progressFill.rectTransform.anchorMax = new Vector2(p, 1);
            progressText.text = currentMs / 60000 + ":" + (currentMs / 1000 % 60).ToString("00");
            if (Time.unscaledTime - lastClockSend > 0.05f) { lastClockSend = Time.unscaledTime; _ = socket.SendAsync(new MediaClockMessage { media_time_ms = currentMs }); }
            if (gradeUntil > 0 && Time.unscaledTime > gradeUntil) { gradeText.text = ""; gradeUntil = 0; }
        }

        private void HandleMessage(SocketEnvelope message)
        {
            if (message.type == "score_event")
            {
                scoreText.text = message.total_score.ToString("N0"); comboText.text = "COMBO  " + message.combo; gradeText.text = string.IsNullOrEmpty(message.grade) ? "" : message.grade.ToUpperInvariant();
                gradeText.color = message.grade == "perfect" ? RuntimeUi.Cyan : message.grade == "super" ? RuntimeUi.Pink : message.grade == "good" ? RuntimeUi.Gold : RuntimeUi.White; gradeUntil = Time.unscaledTime + 0.72f;
            }
            else if (message.type == "phone_disconnected") { gradeText.text = "PHONE DISCONNECTED"; gradeText.color = RuntimeUi.Pink; }
            else if (message.type == "game_over") { started = false; gradeText.text = "ROUTINE COMPLETE"; gradeText.color = RuntimeUi.Gold; }
        }

        private void TogglePause()
        {
            if (!started || video == null) return; paused = !paused;
            if (paused) { video.Pause(); _ = socket.SendAsync(new GameActionMessage { action = "pause" }); }
            else { video.Play(); _ = socket.SendAsync(new GameActionMessage { action = "resume" }); }
        }

        private void OnVideoFinished(VideoPlayer source) { if (socket != null) _ = socket.SendAsync(new MediaClockMessage { media_time_ms = dance.duration_ms }); }
        private void Exit() { app.ShowLibrary(); }
        private void OnDestroy()
        {
            if (app != null) { app.Input.Cancel -= Exit; app.Input.Submit -= TogglePause; }
            if (socket != null) socket.Dispose(); if (video != null) video.Stop(); if (videoTexture != null) videoTexture.Release();
        }
    }

    public sealed class CoachStage3D : MonoBehaviour
    {
        private DanceApiClient api; private string danceId; private VideoPlayer clock; private RawImage sourceVideo; private GameObject coach; private Camera coachCamera;
        private RenderTexture coachTexture; private HumanoidPoseDriver driver; private PoseTimeline timeline; private bool active3D;

        public void Initialize(Canvas canvas, DanceApiClient client, string id, VideoPlayer videoClock, RawImage videoImage)
        {
            api = client; danceId = id; clock = videoClock; sourceVideo = videoImage;
            GameObject prefab = Resources.Load<GameObject>("Coaches/" + id) ?? Resources.Load<GameObject>("Coaches/DefaultCoach"); if (prefab == null) return; active3D = true;
            RawImage stageBackground = RuntimeUi.Panel(canvas.transform, "3DBackground", RuntimeUi.Deep, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            stageBackground.transform.SetSiblingIndex(Mathf.Max(1, sourceVideo.transform.GetSiblingIndex() + 1));
            Texture manualBackground = Resources.Load<Texture>("Backgrounds/" + id) ?? Resources.Load<Texture>("Backgrounds/DefaultBackground");
            if (manualBackground != null) stageBackground.texture = manualBackground; else _ = RuntimeUi.SetTextureAsync(stageBackground, api, api.PosterUrl(danceId));
            RawImage coachSurface = RuntimeUi.Panel(canvas.transform, "3DCoach", Color.white, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            coachSurface.transform.SetSiblingIndex(stageBackground.transform.GetSiblingIndex() + 1); sourceVideo.color = new Color(1, 1, 1, 0.001f);
            int rw = Screen.width >= 3000 ? 3840 : 1920; int rh = Screen.width >= 3000 ? 2160 : 1080;
            coachTexture = new RenderTexture(rw, rh, 24, RenderTextureFormat.ARGB32); coachTexture.Create(); coachSurface.texture = coachTexture;
            GameObject cameraObject = new GameObject("CoachCamera"); cameraObject.transform.SetParent(transform, false); coachCamera = cameraObject.AddComponent<Camera>();
            coachCamera.clearFlags = CameraClearFlags.SolidColor; coachCamera.backgroundColor = new Color(0, 0, 0, 0); coachCamera.targetTexture = coachTexture; coachCamera.fieldOfView = 34f; coachCamera.cullingMask = 1 << 8;
            coach = Instantiate(prefab); coach.name = "DefaultCoach"; coach.transform.SetParent(transform, false); SetLayerRecursive(coach.transform, 8);
            driver = coach.AddComponent<HumanoidPoseDriver>(); FrameCoach(); _ = LoadTimelineAsync();
        }

        private async Task LoadTimelineAsync() { try { timeline = await api.GetAsync<PoseTimeline>(api.PoseTimelineUrl(danceId, 20)); } catch (Exception e) { Debug.LogWarning("3D timeline unavailable: " + e.Message); } }
        private void Update()
        {
            if (!active3D || timeline == null || timeline.frames == null || timeline.frames.Length == 0 || driver == null || clock == null) return;
            driver.Apply(Sample(Mathf.Max(0, (int)(clock.time * 1000.0))).landmarks);
        }

        private PoseTimelineFrame Sample(int t)
        {
            PoseTimelineFrame[] frames = timeline.frames; int lo = 0, hi = frames.Length - 1;
            while (lo < hi) { int mid = (lo + hi) / 2; if (frames[mid].t_ms < t) lo = mid + 1; else hi = mid; }
            int idx = Mathf.Clamp(lo, 0, frames.Length - 1); if (idx > 0 && Mathf.Abs(frames[idx - 1].t_ms - t) < Mathf.Abs(frames[idx].t_ms - t)) idx--; return frames[idx];
        }

        private void FrameCoach()
        {
            Renderer[] renderers = coach.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) { coachCamera.transform.position = new Vector3(0, 1.15f, -4.5f); coachCamera.transform.LookAt(new Vector3(0, 1.1f, 0)); return; }
            Bounds bounds = renderers[0].bounds; for (int i = 1; i < renderers.Length; i++) bounds.Encapsulate(renderers[i].bounds);
            float halfHeight = Mathf.Max(bounds.extents.y, 0.5f); float distance = halfHeight / Mathf.Tan(coachCamera.fieldOfView * 0.5f * Mathf.Deg2Rad) * 1.55f;
            Vector3 center = bounds.center; coachCamera.transform.position = new Vector3(center.x, center.y + halfHeight * 0.02f, center.z - distance); coachCamera.transform.LookAt(center + Vector3.up * halfHeight * 0.02f);
        }

        private static void SetLayerRecursive(Transform root, int layer) { root.gameObject.layer = layer; foreach (Transform child in root) SetLayerRecursive(child, layer); }
        private void OnDestroy() { if (coachTexture != null) coachTexture.Release(); if (coach != null) Destroy(coach); }
    }

    public sealed class HumanoidPoseDriver : MonoBehaviour
    {
        private sealed class BoneMap { public Transform bone; public Transform child; public int a; public int b; public Quaternion restRotation; public Vector3 restDirection; }
        private Animator animator; private readonly List<BoneMap> maps = new List<BoneMap>(); private bool initialized;

        private void Start()
        {
            animator = GetComponentInChildren<Animator>();
            if (animator == null || !animator.isHuman) { Debug.LogWarning("DefaultCoach must use a Humanoid Avatar for pose retargeting."); enabled = false; return; }
            Add(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, 11, 13); Add(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, 13, 15);
            Add(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, 12, 14); Add(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, 14, 16);
            Add(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, 23, 25); Add(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, 25, 27);
            Add(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, 24, 26); Add(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, 26, 28);
            initialized = maps.Count > 0;
        }

        private void Add(HumanBodyBones boneId, HumanBodyBones childId, int a, int b)
        {
            Transform bone = animator.GetBoneTransform(boneId); Transform child = animator.GetBoneTransform(childId); if (bone == null || child == null) return;
            maps.Add(new BoneMap { bone = bone, child = child, a = a, b = b, restRotation = bone.rotation, restDirection = (child.position - bone.position).normalized });
        }

        public void Apply(Landmark[] pose)
        {
            if (!initialized || pose == null || pose.Length < 29) return;
            foreach (BoneMap map in maps)
            {
                Landmark a = pose[map.a]; Landmark b = pose[map.b]; if (a == null || b == null || a.v < 0.10f || b.v < 0.10f) continue;
                Vector3 target = new Vector3(b.x - a.x, -(b.y - a.y), -(b.z - a.z)); if (target.sqrMagnitude < 0.000001f) continue; target.Normalize();
                Quaternion desired = Quaternion.FromToRotation(map.restDirection, target) * map.restRotation; map.bone.rotation = Quaternion.Slerp(map.bone.rotation, desired, 0.42f);
            }
        }
    }
}
