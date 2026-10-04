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
    [Serializable] public class DanceListItem { public string dance_id; public string title; public int duration_ms; public string difficulty; public string created_at; public bool has_video; public bool has_poster; public string preview_mode; public DanceTheme theme; public int coach_count = 1; }
    [Serializable] public class DanceDetail { public string dance_id; public string title; public int version; public int duration_ms; public string skeleton_format; public string preview_mode; public string difficulty; public bool mirror_mode; public string created_at; public int num_frames; public int num_events; public bool has_poster; public DanceTheme theme; public int coach_count = 1; }
    [Serializable] public class SessionCreateRequest { public string dance_id; }
    [Serializable] public class GameSession { public string session_id; public string dance_id; public string connect_url; public string qr_data; }
    [Serializable] public class PlayerStatus { public string player_id; public int slot; public int coach_index; public bool ready; public float tracking_score; public bool active; }
    [Serializable] public class SessionStatus { public string status; public bool phone_connected; public bool calibrated; public int coach_count = 1; public PlayerStatus[] players; }
    [Serializable] public class SessionAssignRequest { public string player_id; public int coach_index; }
    [Serializable] public class SessionAssignResponse { public string status; public string player_id; public int player_slot; public int coach_index; public PlayerStatus[] players; }
    [Serializable] public class Landmark { public float x; public float y; public float z; public float v; }
    [Serializable] public class PoseTimelineFrame { public int t_ms; public Landmark[] landmarks; }
    [Serializable] public class PoseTimeline { public string dance_id; public int fps; public int duration_ms; public string space; public PoseTimelineFrame[] frames; }
    [Serializable] public class MotionHint { public int joint; public float dx; public float dy; public float magnitude; }
    [Serializable] public class MovePreview { public int t_ms; public int move_index; public int cue_index; public float motion; public Landmark[] start_landmarks; public Landmark[] landmarks; public MotionHint[] motion_hints; }
    [Serializable] public class CoachCueTrack { public int coach_index; public MovePreview[] cues; }
    [Serializable] public class CoachInfo { public int coach_index; public string label; public string preview_path; public float coverage; public float avg_x; }
    [Serializable] public class PlaybackData
    {
        public string dance_id;
        public int duration_ms;
        public bool mirror_mode;
        public float tempo;
        public int[] beat_ms;
        public int[] strong_beat_ms;
        public MovePreview[] move_previews;
        public int move_count;
        public int coach_count = 1;
        public CoachCueTrack[] coach_cues;
        public CoachInfo[] coaches;
    }
    [Serializable] public class GameResults { public int total_score; public int max_combo; public float accuracy_arms; public float accuracy_legs; public float accuracy_torso; }
    [Serializable] public class SocketEnvelope { public string type; public string grade; public int timestamp_ms; public int score; public int total_score; public int combo; public float similarity; public string hold_state; public float timing_offset_ms; public bool tracking_lost; public int move_index; public int move_count; public bool is_move_grade; public string player_id; public int player_slot; public int coach_index; public Landmark[] player_pose; public PlayerStatus[] players; public int coach_count; public GameResults results; }
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
        public static bool Enable3DCoach
        {
            get
            {
                string env = Environment.GetEnvironmentVariable("DANCE_ENABLE_3D") ?? "";
                if (env == "1" || env.Equals("true", StringComparison.OrdinalIgnoreCase)) return true;
                return PlayerPrefs.GetInt("dance_enable_3d", 0) == 1;
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
        public string CoachPreviewUrl(string danceId, int coachIndex) { return BaseUrl + "/api/dances/" + danceId + "/coaches/" + coachIndex + "/preview"; }
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

    public sealed class PosePreviewGraphic : MaskableGraphic
    {
        private static readonly int[,] Bones = new int[,]
        {
            {11,12},{11,13},{13,15},{12,14},{14,16},
            {11,23},{12,24},{23,24},{23,25},{25,27},
            {24,26},{26,28}
        };
        private Landmark[] pose = Array.Empty<Landmark>();
        private Landmark[] ghostPose = Array.Empty<Landmark>();
        private MotionHint[] motionHints = Array.Empty<MotionHint>();
        private bool primary;
        private Color32 tint = new Color32(255, 64, 226, 255);

        public void SetPose(Landmark[] value, bool isPrimary)
        {
            SetPose(value, isPrimary, isPrimary ? new Color32(255,64,226,255) : new Color32(142,96,255,235), null);
        }

        public void SetPose(Landmark[] value, bool isPrimary, Color32 tintColor)
        {
            SetPose(value, isPrimary, tintColor, null);
        }

        public void SetPose(Landmark[] value, bool isPrimary, Color32 tintColor, MotionHint[] hints)
        {
            ghostPose = Array.Empty<Landmark>();
            pose = value ?? Array.Empty<Landmark>();
            motionHints = hints ?? Array.Empty<MotionHint>();
            primary = isPrimary;
            tint = tintColor;
            color = Color.white;
            SetVerticesDirty();
        }

        public void SetMotionPreview(Landmark[] startPose, Landmark[] targetPose, bool isPrimary, Color32 tintColor, MotionHint[] hints)
        {
            ghostPose = startPose ?? Array.Empty<Landmark>();
            pose = targetPose ?? Array.Empty<Landmark>();
            motionHints = hints ?? Array.Empty<MotionHint>();
            primary = isPrimary;
            tint = tintColor;
            color = Color.white;
            SetVerticesDirty();
        }

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            if (pose == null || pose.Length < 29) return;

            List<int> joints = new List<int> { 0,11,12,13,14,15,16,23,24,25,26,27,28 };
            float minX = 10f, minY = 10f, maxX = -10f, maxY = -10f;
            foreach (int idx in joints)
            {
                if (idx >= pose.Length) continue;
                Landmark lm = pose[idx];
                if (lm == null || lm.v < 0.10f) continue;
                minX = Mathf.Min(minX, lm.x); maxX = Mathf.Max(maxX, lm.x);
                minY = Mathf.Min(minY, lm.y); maxY = Mathf.Max(maxY, lm.y);
            }
            if (ghostPose != null && ghostPose.Length >= 29)
            {
                foreach (int idx in joints)
                {
                    if (idx >= ghostPose.Length) continue;
                    Landmark lm = ghostPose[idx];
                    if (lm == null || lm.v < 0.10f) continue;
                    minX = Mathf.Min(minX, lm.x); maxX = Mathf.Max(maxX, lm.x);
                    minY = Mathf.Min(minY, lm.y); maxY = Mathf.Max(maxY, lm.y);
                }
            }
            if (maxX <= minX || maxY <= minY) return;

            Rect rect = rectTransform.rect;
            float pad = Mathf.Min(rect.width, rect.height) * 0.13f;
            float scale = Mathf.Min((rect.width - pad * 2f) / Mathf.Max(maxX - minX, 0.001f),
                                    (rect.height - pad * 2f) / Mathf.Max(maxY - minY, 0.001f));
            Vector2 center = rect.center;

            Vector2 MapFrom(Landmark[] source, int idx)
            {
                Landmark lm = source[idx];
                float x = (lm.x - (minX + maxX) * 0.5f) * scale;
                float y = -(lm.y - (minY + maxY) * 0.5f) * scale;
                return center + new Vector2(x, y);
            }
            Vector2 Map(int idx) { return MapFrom(pose, idx); }

            float lineWidth = primary ? 8.5f : 7f;
            Color32 outer = new Color32(tint.r, tint.g, tint.b, (byte)(primary ? 245 : 220));
            Color32 glow = new Color32(tint.r, tint.g, tint.b, (byte)(primary ? 115 : 75));
            Color32 inner = new Color32(255,255,255,(byte)(primary ? 255 : 235));

            int boneCount = Bones.GetLength(0);
            if (ghostPose != null && ghostPose.Length >= 29)
            {
                Color32 ghost = new Color32(tint.r, tint.g, tint.b, 64);
                for (int i = 0; i < boneCount; i++)
                {
                    int a = Bones[i,0], b = Bones[i,1];
                    if (ghostPose[a] == null || ghostPose[b] == null || Mathf.Min(ghostPose[a].v, ghostPose[b].v) < 0.10f) continue;
                    AddLine(vh, MapFrom(ghostPose, a), MapFrom(ghostPose, b), primary ? 5.5f : 4.5f, ghost);
                }
            }

            for (int i = 0; i < boneCount; i++)
            {
                int a = Bones[i,0], b = Bones[i,1];
                if (pose[a] == null || pose[b] == null || Mathf.Min(pose[a].v, pose[b].v) < 0.10f) continue;
                AddLine(vh, Map(a), Map(b), lineWidth + 12f, glow);
                AddLine(vh, Map(a), Map(b), lineWidth + 4f, outer);
                AddLine(vh, Map(a), Map(b), Mathf.Max(3f, lineWidth * .42f), inner);
            }

            Vector2 head;
            if (pose.Length > 0 && pose[0] != null && pose[0].v >= 0.10f)
                head = Map(0);
            else
                head = (Map(11) + Map(12)) * .5f + Vector2.up * (Mathf.Abs(Map(11).x - Map(12).x) * .82f);
            AddCircle(vh, head, primary ? 15f : 13f, outer, 14);
            AddCircle(vh, head, primary ? 9f : 8f, inner, 14);

            foreach (int idx in new[] { 15,16,27,28 })
            {
                if (pose[idx] == null || pose[idx].v < 0.10f) continue;
                AddCircle(vh, Map(idx), primary ? 8f : 6.5f, outer, 10);
            }

            if (motionHints != null && motionHints.Length > 0)
            {
                foreach (MotionHint hint in motionHints)
                {
                    if (hint == null || hint.joint < 0 || hint.joint >= pose.Length) continue;
                    Landmark lm = pose[hint.joint];
                    if (lm == null || lm.v < 0.10f) continue;

                    Vector2 end = Map(hint.joint);
                    Vector2 start;
                    if (ghostPose != null && ghostPose.Length > hint.joint && ghostPose[hint.joint] != null && ghostPose[hint.joint].v >= 0.10f)
                    {
                        start = MapFrom(ghostPose, hint.joint);
                    }
                    else
                    {
                        Vector2 delta = new Vector2(hint.dx, -hint.dy) * scale;
                        float maxLen = Mathf.Min(rect.width, rect.height) * .34f;
                        if (delta.magnitude > maxLen) delta = delta.normalized * maxLen;
                        if (delta.magnitude < 14f) continue;
                        start = end - delta;
                    }
                    if ((end - start).magnitude < 14f) continue;
                    AddArrow(vh, start, end, primary ? 6.5f : 5f, outer);
                }
            }
        }

        private static void AddLine(VertexHelper vh, Vector2 a, Vector2 b, float width, Color32 color)
        {
            Vector2 delta = b - a;
            if (delta.sqrMagnitude < 0.0001f) return;
            Vector2 dir = delta.normalized;
            Vector2 n = new Vector2(-dir.y, dir.x) * (width * 0.5f);
            int start = vh.currentVertCount;
            vh.AddVert(a - n, color, Vector2.zero);
            vh.AddVert(a + n, color, Vector2.zero);
            vh.AddVert(b + n, color, Vector2.zero);
            vh.AddVert(b - n, color, Vector2.zero);
            vh.AddTriangle(start, start + 1, start + 2);
            vh.AddTriangle(start, start + 2, start + 3);
        }

        private static void AddCircle(VertexHelper vh, Vector2 center, float radius, Color32 color, int segments)
        {
            int centerIndex = vh.currentVertCount;
            vh.AddVert(center, color, new Vector2(.5f,.5f));
            for (int i = 0; i <= segments; i++)
            {
                float a = Mathf.PI * 2f * i / segments;
                Vector2 p = center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius;
                vh.AddVert(p, color, Vector2.zero);
            }
            for (int i = 0; i < segments; i++)
                vh.AddTriangle(centerIndex, centerIndex + i + 1, centerIndex + i + 2);
        }

        private static void AddArrow(VertexHelper vh, Vector2 start, Vector2 end, float width, Color32 color)
        {
            AddLine(vh, start, end, width, color);
            Vector2 delta = end - start;
            if (delta.sqrMagnitude < 0.0001f) return;
            Vector2 dir = delta.normalized;
            Vector2 side = new Vector2(-dir.y, dir.x);
            float head = Mathf.Clamp(delta.magnitude * .22f, 12f, 26f);
            Vector2 left = end - dir * head + side * head * .46f;
            Vector2 right = end - dir * head - side * head * .46f;
            AddLine(vh, left, end, width, color);
            AddLine(vh, right, end, width, color);
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
        public void ShowCoachSelect(GameSession session, PlayerStatus[] players, int coachCount) { CurrentSession = session; NewScreen<CoachSelectScreen>("CoachSelectScreen").Initialize(this, SelectedDance, session, players, coachCount); }
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
            selectedMeta.text = dance.difficulty.ToUpperInvariant() + "     " + minutes + ":" + seconds.ToString("00") +
                "     1–4 PLAYERS" + (dance.coach_count > 1 ? "     " + dance.coach_count + " COACHES" : "");
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
        private DanceFlowApp app; private DanceListItem dance; private GameSession session; private Text status; private RawImage qr; private bool ready; private int generation; private SessionStatus lastStatus;
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
                    lastStatus = state;
                    if (state.phone_connected)
                    {
                        int activePlayers = state.players == null ? 0 : state.players.Count(p => p.active);
                        if (state.calibrated)
                        {
                            ready = true;
                            status.text = "✓ PHONE READY     " + Mathf.Max(1, activePlayers) + " PLAYER" + (activePlayers == 1 ? "" : "S") + " TRACKED     PRESS A / ENTER";
                            break;
                        }
                        status.text = "✓ PHONE CONNECTED     " + Mathf.Max(1, activePlayers) + " PLAYER" + (activePlayers == 1 ? "" : "S") + "     CALIBRATING…";
                    }
                    await Task.Delay(550);
                }
            }
            catch (Exception e) { if (status != null) status.text = "SESSION ERROR: " + e.Message; }
        }

        private void TryStart()
        {
            if (!ready || session == null) return;
            PlayerStatus[] players = lastStatus != null && lastStatus.players != null
                ? lastStatus.players.Where(p => p.active).OrderBy(p => p.slot).ToArray()
                : Array.Empty<PlayerStatus>();
            if (players.Length == 0)
                players = new[] { new PlayerStatus { player_id = "p0", slot = 0, coach_index = 0, ready = true, active = true } };
            int coachCount = lastStatus != null ? Mathf.Max(1, lastStatus.coach_count) : Mathf.Max(1, dance.coach_count);
            if (coachCount > 1 || players.Length > 1) app.ShowCoachSelect(session, players, coachCount);
            else app.ShowGameplay(session);
        }
        private void Back() { app.ShowLibrary(); }
        private void OnDestroy() { generation++; if (app != null) { app.Input.Cancel -= Back; app.Input.Submit -= TryStart; } }
    }

    public sealed class CoachSelectScreen : MonoBehaviour
    {
        private DanceFlowApp app;
        private DanceListItem dance;
        private GameSession session;
        private PlayerStatus[] players;
        private Canvas canvas;
        private Text title;
        private Text playerLabel;
        private Text hint;
        private readonly List<RawImage> cards = new List<RawImage>();
        private readonly List<Text> cardLabels = new List<Text>();
        private int playerCursor;
        private int selectedCoach;
        private int coachCount = 1;
        private bool committing;

        private static Color32 SlotColor(int slot)
        {
            Color32[] colors = {
                new Color32(108,255,85,255),
                new Color32(188,103,255,255),
                new Color32(255,204,66,255),
                new Color32(70,229,255,255),
            };
            return colors[Mathf.Abs(slot) % colors.Length];
        }

        private static Color32 CoachColor(int coachIndex)
        {
            Color32[] colors = {
                new Color32(108,255,85,255),
                new Color32(188,103,255,255),
                new Color32(255,204,66,255),
                new Color32(70,229,255,255),
            };
            return colors[Mathf.Abs(coachIndex) % colors.Length];
        }

        public void Initialize(DanceFlowApp value, DanceListItem selectedDance, GameSession currentSession, PlayerStatus[] detectedPlayers, int detectedCoachCount)
        {
            app = value;
            dance = selectedDance;
            session = currentSession;
            coachCount = Mathf.Clamp(Mathf.Max(1, detectedCoachCount), 1, 4);
            players = detectedPlayers != null && detectedPlayers.Length > 0
                ? detectedPlayers.OrderBy(p => p.slot).ToArray()
                : new[] { new PlayerStatus { player_id = "p0", slot = 0, coach_index = 0, ready = true, active = true } };
            BuildUi();
            app.Input.Left += Prev;
            app.Input.Right += Next;
            app.Input.Submit += Confirm;
            app.Input.Cancel += Back;
            SetPlayer(0);
        }

        private void BuildUi()
        {
            canvas = RuntimeUi.CreateCanvas(transform, "Coach Select", 0);
            RawImage bg = RuntimeUi.Panel(canvas.transform, "Background", RuntimeUi.Deep, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            if (dance.has_poster) _ = RuntimeUi.SetTextureAsync(bg, app.Api, app.Api.PosterUrl(dance.dance_id));
            RuntimeUi.Panel(canvas.transform, "Wash", new Color(0.18f,0.02f,0.35f,0.78f), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            RuntimeUi.Label(canvas.transform, "Brand", "DANCEFLOW", 88, TextAnchor.MiddleLeft, RuntimeUi.White,
                new Vector2(0,1), new Vector2(0,1), new Vector2(65,-145), new Vector2(700,-45));
            title = RuntimeUi.Label(canvas.transform, "Title", "SELECT YOUR COACH", 82, TextAnchor.MiddleCenter, RuntimeUi.White,
                new Vector2(.18f,1), new Vector2(.82f,1), new Vector2(0,-210), new Vector2(0,-80));
            playerLabel = RuntimeUi.Label(canvas.transform, "Player", "", 42, TextAnchor.MiddleCenter, RuntimeUi.Cyan,
                new Vector2(.25f,1), new Vector2(.75f,1), new Vector2(0,-300), new Vector2(0,-220));

            int visibleCoachCount = coachCount;
            float cardWidth = visibleCoachCount == 1 ? 840f : visibleCoachCount == 2 ? 730f : visibleCoachCount == 3 ? 610f : 520f;
            float gap = 55f;
            float total = visibleCoachCount * cardWidth + (visibleCoachCount - 1) * gap;
            float left = -total * .5f;

            for (int i = 0; i < visibleCoachCount; i++)
            {
                float x0 = left + i * (cardWidth + gap);
                RawImage card = RuntimeUi.Panel(canvas.transform, "CoachCard" + i, new Color(0.08f,0.03f,0.16f,.92f),
                    new Vector2(.5f,.5f), new Vector2(.5f,.5f),
                    new Vector2(x0,-650), new Vector2(x0 + cardWidth, 420));
                RawImage art = RuntimeUi.Panel(card.transform, "Art", Color.white, Vector2.zero, Vector2.one,
                    new Vector2(18,105), new Vector2(-18,-18));
                if (dance.has_poster) _ = RuntimeUi.SetTextureAsync(art, app.Api, app.Api.PosterUrl(dance.dance_id));
                _ = RuntimeUi.SetTextureAsync(art, app.Api, app.Api.CoachPreviewUrl(dance.dance_id, i));
                Text label = RuntimeUi.Label(card.transform, "Label", "COACH " + (i + 1), 38, TextAnchor.MiddleCenter, RuntimeUi.White,
                    new Vector2(0,0), new Vector2(1,0), new Vector2(15,18), new Vector2(-15,95));
                cards.Add(card);
                cardLabels.Add(label);
            }

            hint = RuntimeUi.Label(canvas.transform, "Hint", "◀  ▶  CHOOSE        A / ENTER  CONFIRM", 27, TextAnchor.MiddleCenter,
                new Color(1,1,1,.78f), new Vector2(0,0), new Vector2(1,0), new Vector2(0,35), new Vector2(0,95));
        }

        private void SetPlayer(int index)
        {
            playerCursor = Mathf.Clamp(index, 0, players.Length - 1);
            PlayerStatus p = players[playerCursor];
            selectedCoach = Mathf.Clamp(p.coach_index, 0, cards.Count - 1);
            Color32 color = SlotColor(p.slot);
            playerLabel.text = "PLAYER " + (p.slot + 1) + "  •  CHOOSE THE DANCER YOU WANT TO FOLLOW";
            playerLabel.color = color;
            UpdateCards();
        }

        private void UpdateCards()
        {
            for (int i = 0; i < cards.Count; i++)
            {
                bool active = i == selectedCoach;
                Color32 playerColor = SlotColor(players[playerCursor].slot);
                cards[i].color = active
                    ? new Color(playerColor.r / 255f * .48f, playerColor.g / 255f * .22f, playerColor.b / 255f * .48f, .98f)
                    : new Color(.055f,.025f,.12f,.90f);
                cards[i].rectTransform.localScale = active ? Vector3.one * 1.055f : Vector3.one;
                cardLabels[i].text = active ? "✓  COACH " + (i + 1) : "COACH " + (i + 1);
                cardLabels[i].color = active ? Color.white : new Color(.82f,.80f,.91f,1);
            }
            hint.text = "PLAYER " + (players[playerCursor].slot + 1) + "     ◀  ▶  CHOOSE        A / ENTER  CONFIRM";
        }

        private void Prev() { if (!committing) { selectedCoach = (selectedCoach - 1 + cards.Count) % cards.Count; UpdateCards(); } }
        private void Next() { if (!committing) { selectedCoach = (selectedCoach + 1) % cards.Count; UpdateCards(); } }
        private void Confirm() { if (!committing) _ = ConfirmAsync(); }

        private async Task ConfirmAsync()
        {
            committing = true;
            PlayerStatus p = players[playerCursor];
            hint.text = "SAVING PLAYER " + (p.slot + 1) + "…";
            try
            {
                await app.Api.PostAsync<SessionAssignRequest, SessionAssignResponse>(
                    "/api/session/" + session.session_id + "/assign",
                    new SessionAssignRequest { player_id = p.player_id, coach_index = selectedCoach }
                );
                p.coach_index = selectedCoach;
                if (playerCursor + 1 < players.Length)
                {
                    SetPlayer(playerCursor + 1);
                    committing = false;
                }
                else
                {
                    app.ShowGameplay(session);
                }
            }
            catch (Exception e)
            {
                hint.text = "ASSIGNMENT ERROR: " + e.Message;
                committing = false;
            }
        }

        private void Back() { app.ShowConnect(dance); }

        private void OnDestroy()
        {
            if (app == null || app.Input == null) return;
            app.Input.Left -= Prev;
            app.Input.Right -= Next;
            app.Input.Submit -= Confirm;
            app.Input.Cancel -= Back;
        }
    }

    public sealed class GameplayScreen : MonoBehaviour
    {
        private sealed class PlayerHud
        {
            public GameObject root;
            public RectTransform rect;
            public PosePreviewGraphic mirror;
            public Text playerLabel;
            public Text grade;
            public Text score;
            public Text combo;
            public string playerId;
            public int slot;
            public int coachIndex;
            public float gradeUntil;
        }

        private DanceFlowApp app;
        private DanceListItem dance;
        private GameSession session;
        private GameSocketClient socket;
        private VideoPlayer video;
        private RenderTexture videoTexture;
        private RawImage videoSurface;
        private Text progressText;
        private RawImage progressFill;
        private Text countdownText;
        private Text systemText;
        private bool started;
        private bool paused;
        private float lastClockSend;
        private PlaybackData playback;
        private SessionStatus sessionStatus;

        private readonly PlayerHud[] playerHuds = new PlayerHud[4];
        private readonly PosePreviewGraphic[] cueGraphics = new PosePreviewGraphic[4];
        private readonly RawImage[] cueProgress = new RawImage[4];
        private readonly int[] lastCueIndex = { -1, -1, -1, -1 };

        private static Color32 SlotColor(int slot)
        {
            Color32[] colors = {
                new Color32(108,255,85,255),
                new Color32(188,103,255,255),
                new Color32(255,204,66,255),
                new Color32(70,229,255,255),
            };
            return colors[Mathf.Abs(slot) % colors.Length];
        }

        public void Initialize(DanceFlowApp value, DanceListItem selected, GameSession currentSession)
        {
            app = value;
            dance = selected;
            session = currentSession;
            BuildUi();
            app.Input.Cancel += Exit;
            app.Input.Submit += TogglePause;
            _ = StartGameAsync();
        }

        private void BuildUi()
        {
            Canvas canvas = RuntimeUi.CreateCanvas(transform, "Gameplay", 0);
            videoSurface = RuntimeUi.Panel(canvas.transform, "Video", Color.white, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            int rw = Screen.width >= 3000 ? 3840 : 1920;
            int rh = Screen.width >= 3000 ? 2160 : 1080;
            videoTexture = new RenderTexture(rw, rh, 0, RenderTextureFormat.ARGB32);
            video = gameObject.AddComponent<VideoPlayer>();
            video.playOnAwake = false;
            video.url = app.Api.VideoUrl(dance.dance_id);
            video.renderMode = VideoRenderMode.RenderTexture;
            video.targetTexture = videoTexture;
            video.audioOutputMode = VideoAudioOutputMode.Direct;
            video.skipOnDrop = false;
            videoSurface.texture = videoTexture;
            video.loopPointReached += OnVideoFinished;

            RuntimeUi.Panel(canvas.transform, "Vignette", new Color(0.02f,0.01f,0.06f,0.10f), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            // Song information stays edge-bound. The middle of the TV is reserved
            // for the dancers, matching the supplied Just Dance reference.
            RawImage song = RuntimeUi.Panel(canvas.transform, "SongPanel", new Color(0.025f,0.015f,0.09f,0.76f),
                new Vector2(0,1), new Vector2(0,1), new Vector2(45,-205), new Vector2(1220,-42));
            RawImage cover = RuntimeUi.Panel(song.transform, "Cover", Color.white, new Vector2(0,0), new Vector2(0,1),
                new Vector2(12,12), new Vector2(172,-12));
            if (dance.has_poster) _ = RuntimeUi.SetTextureAsync(cover, app.Api, app.Api.PosterUrl(dance.dance_id));
            RuntimeUi.Label(song.transform, "Title", dance.title, 48, TextAnchor.MiddleLeft, RuntimeUi.White,
                new Vector2(0,0.46f), new Vector2(1,1), new Vector2(200,0), new Vector2(-25,-4));
            progressText = RuntimeUi.Label(song.transform, "Time", "0:00", 27, TextAnchor.MiddleRight, new Color(.9f,.9f,.98f),
                new Vector2(.72f,0), new Vector2(1,.45f), new Vector2(0,6), new Vector2(-25,0));
            RawImage bar = RuntimeUi.Panel(song.transform, "ProgressBar", new Color(1,1,1,.18f), new Vector2(0,0), new Vector2(1,0),
                new Vector2(200,27), new Vector2(-145,43));
            progressFill = RuntimeUi.Panel(bar.transform, "Fill", RuntimeUi.Pink, Vector2.zero, new Vector2(0,1), Vector2.zero, Vector2.zero);

            BuildPlayerCards(canvas);
            BuildCueStrip(canvas);

            // Thin song timeline; it stops before the bottom-right pictograms.
            RawImage timeline = RuntimeUi.Panel(canvas.transform, "Timeline", new Color(0.025f,0.015f,0.08f,0.52f),
                new Vector2(.07f,0), new Vector2(.74f,0), new Vector2(0,34), new Vector2(0,105));
            RuntimeUi.Panel(timeline.transform, "Line", new Color(.85f,.95f,1,.68f),
                new Vector2(.03f,.5f), new Vector2(.97f,.5f), new Vector2(0,-2), new Vector2(0,2));

            countdownText = RuntimeUi.Label(canvas.transform, "Countdown", "", 168, TextAnchor.MiddleCenter, RuntimeUi.White,
                new Vector2(.35f,.38f), new Vector2(.65f,.67f), Vector2.zero, Vector2.zero);
            systemText = RuntimeUi.Label(canvas.transform, "System", "", 34, TextAnchor.MiddleCenter, RuntimeUi.Pink,
                new Vector2(.30f,1), new Vector2(.70f,1), new Vector2(0,-315), new Vector2(0,-255));

            CoachStage3D coach3D = gameObject.AddComponent<CoachStage3D>();
            coach3D.Initialize(canvas, app.Api, dance.dance_id, video, videoSurface);
        }

        private void BuildPlayerCards(Canvas canvas)
        {
            const float cardWidth = 420f;
            for (int i = 0; i < playerHuds.Length; i++)
            {
                RawImage panel = RuntimeUi.Panel(canvas.transform, "PlayerHud" + i, new Color(0.025f,0.018f,0.09f,.66f),
                    new Vector2(0,1), new Vector2(0,1), new Vector2(0,-178), new Vector2(cardWidth,-38));
                panel.gameObject.SetActive(false);

                RuntimeUi.Panel(panel.transform, "Accent", SlotColor(i), new Vector2(0,0), new Vector2(0,1),
                    new Vector2(0,0), new Vector2(7,0));
                RectTransform mirrorRect = RuntimeUi.Rect(panel.transform, "LiveMirror", Vector2.zero, Vector2.zero,
                    new Vector2(13,13), new Vector2(120,127));
                PosePreviewGraphic mirror = mirrorRect.gameObject.AddComponent<PosePreviewGraphic>();
                mirror.raycastTarget = false;

                Text playerLabel = RuntimeUi.Label(panel.transform, "Player", "P" + (i + 1), 22, TextAnchor.UpperLeft,
                    RuntimeUi.White, Vector2.zero, Vector2.one, new Vector2(140,-4), new Vector2(-10,-12));
                Text grade = RuntimeUi.Label(panel.transform, "Grade", "", 36, TextAnchor.MiddleLeft,
                    RuntimeUi.Cyan, Vector2.zero, Vector2.one, new Vector2(140,33), new Vector2(-10,-43));
                Text score = RuntimeUi.Label(panel.transform, "Score", "0", 28, TextAnchor.LowerLeft,
                    RuntimeUi.White, Vector2.zero, Vector2.one, new Vector2(140,9), new Vector2(-150,-8));
                Text combo = RuntimeUi.Label(panel.transform, "Combo", "", 20, TextAnchor.LowerRight,
                    new Color(.93f,.90f,1,1), Vector2.zero, Vector2.one, new Vector2(250,9), new Vector2(-12,-8));

                playerHuds[i] = new PlayerHud {
                    root = panel.gameObject,
                    rect = panel.rectTransform,
                    mirror = mirror,
                    playerLabel = playerLabel,
                    grade = grade,
                    score = score,
                    combo = combo,
                    slot = i,
                    coachIndex = 0,
                    playerId = "p" + i,
                };
            }
        }

        private void LayoutPlayerCards(int activeCount)
        {
            activeCount = Mathf.Clamp(activeCount, 1, playerHuds.Length);
            const float cardWidth = 420f;
            const float gap = 18f;
            const float centerX = 2120f;
            float total = activeCount * cardWidth + (activeCount - 1) * gap;
            float left = centerX - total * .5f;
            int visibleIndex = 0;
            foreach (PlayerHud hud in playerHuds)
            {
                if (hud == null || !hud.root.activeSelf) continue;
                float x0 = left + visibleIndex * (cardWidth + gap);
                hud.rect.offsetMin = new Vector2(x0, -178);
                hud.rect.offsetMax = new Vector2(x0 + cardWidth, -38);
                visibleIndex++;
            }
        }

        private void BuildCueStrip(Canvas canvas)
        {
            // Match the reference game: compact, color-coded pictograms live on
            // the bottom-right edge. They are not permanently visible.
            for (int i = 0; i < cueGraphics.Length; i++)
            {
                float right = -50f - i * 178f;
                RawImage cueRoot = RuntimeUi.Panel(canvas.transform, "Cue" + i, new Color(0.01f,0.01f,0.03f,.10f),
                    new Vector2(1,0), new Vector2(1,0), new Vector2(right - 158,48), new Vector2(right,238));
                cueRoot.gameObject.SetActive(false);

                RectTransform glyphRect = RuntimeUi.Rect(cueRoot.transform, "Glyph", Vector2.zero, Vector2.one,
                    new Vector2(4,18), new Vector2(-4,-10));
                PosePreviewGraphic glyph = glyphRect.gameObject.AddComponent<PosePreviewGraphic>();
                glyph.raycastTarget = false;
                cueGraphics[i] = glyph;

                RawImage baseLine = RuntimeUi.Panel(cueRoot.transform, "BaseLine", new Color(1,1,1,.20f),
                    new Vector2(.10f,0), new Vector2(.90f,0), new Vector2(0,7), new Vector2(0,12));
                RawImage fill = RuntimeUi.Panel(baseLine.transform, "CueProgress", Color.white,
                    Vector2.zero, new Vector2(1,1), Vector2.zero, Vector2.zero);
                cueProgress[i] = fill;
            }
        }

        private async Task LoadStateAsync()
        {
            try
            {
                sessionStatus = await app.Api.GetAsync<SessionStatus>("/api/session/" + session.session_id + "/status");
                ApplyPlayers(sessionStatus != null ? sessionStatus.players : null);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Session state unavailable: " + e.Message);
                ApplyPlayers(null);
            }

            try
            {
                playback = await app.Api.GetAsync<PlaybackData>("/api/dances/" + dance.dance_id + "/playback");
                UpdateCueStrip(0);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Pictograms unavailable: " + e.Message);
            }
        }

        private void ApplyPlayers(PlayerStatus[] players)
        {
            PlayerStatus[] source = players != null
                ? players.Where(p => p != null && p.active).OrderBy(p => p.slot).Take(4).ToArray()
                : Array.Empty<PlayerStatus>();
            if (source.Length == 0)
                source = new[] { new PlayerStatus { player_id = "p0", slot = 0, coach_index = 0, active = true, ready = true } };

            for (int i = 0; i < playerHuds.Length; i++)
            {
                bool active = i < source.Length;
                PlayerHud hud = playerHuds[i];
                hud.root.SetActive(active);
                if (!active) continue;

                PlayerStatus p = source[i];
                hud.playerId = string.IsNullOrEmpty(p.player_id) ? "p" + i : p.player_id;
                hud.slot = p.slot;
                hud.coachIndex = p.coach_index;
                Color32 playerColor = SlotColor(hud.slot);
                Color32 coachColor = CoachColor(hud.coachIndex);
                hud.playerLabel.text = "P" + (hud.slot + 1) + "   COACH " + (hud.coachIndex + 1);
                hud.playerLabel.color = coachColor;
                hud.grade.color = coachColor;
                if (hud.mirror != null) hud.mirror.SetPose(Array.Empty<Landmark>(), true, playerColor);
            }
            LayoutPlayerCards(source.Length);
        }

        private PlayerHud FindHud(SocketEnvelope message)
        {
            if (!string.IsNullOrEmpty(message.player_id))
            {
                foreach (PlayerHud hud in playerHuds)
                    if (hud != null && hud.root.activeSelf && hud.playerId == message.player_id) return hud;
            }
            int slot = Mathf.Clamp(message.player_slot, 0, playerHuds.Length - 1);
            return playerHuds[slot];
        }

        private CoachCueTrack CueTrack(int coachIndex)
        {
            if (playback == null || playback.coach_cues == null) return null;
            foreach (CoachCueTrack track in playback.coach_cues)
                if (track != null && track.coach_index == coachIndex) return track;
            return playback.coach_cues.Length > 0 ? playback.coach_cues[0] : null;
        }

        private void UpdateCueStrip(int mediaMs)
        {
            if (playback == null || playback.coach_cues == null) return;

            int coachCount = Mathf.Clamp(playback.coach_count > 0 ? playback.coach_count : playback.coach_cues.Length, 1, cueGraphics.Length);
            const int leadWindowMs = 1900;
            for (int coach = 0; coach < cueGraphics.Length; coach++)
            {
                if (cueGraphics[coach] == null) continue;
                GameObject cueRoot = cueGraphics[coach].transform.parent.gameObject;
                if (coach >= coachCount)
                {
                    cueRoot.SetActive(false);
                    continue;
                }

                CoachCueTrack track = CueTrack(coach);
                if (track == null || track.cues == null || track.cues.Length == 0)
                {
                    cueRoot.SetActive(false);
                    continue;
                }

                int idx = 0;
                while (idx < track.cues.Length && track.cues[idx].t_ms <= mediaMs + 90) idx++;
                if (idx >= track.cues.Length)
                {
                    cueRoot.SetActive(false);
                    continue;
                }

                MovePreview cue = track.cues[idx];
                int untilCue = cue.t_ms - mediaMs;
                bool visible = untilCue >= 0 && untilCue <= leadWindowMs;
                cueRoot.SetActive(visible);
                if (!visible) continue;

                if (lastCueIndex[coach] != idx)
                {
                    lastCueIndex[coach] = idx;
                    cueGraphics[coach].SetMotionPreview(
                        cue.start_landmarks ?? Array.Empty<Landmark>(),
                        cue.landmarks ?? Array.Empty<Landmark>(),
                        true,
                        CoachColor(coach),
                        cue.motion_hints
                    );
                }

                float remain = Mathf.Clamp01(untilCue / (float)leadWindowMs);
                if (cueProgress[coach] != null)
                {
                    cueProgress[coach].color = CoachColor(coach);
                    cueProgress[coach].rectTransform.anchorMax = new Vector2(remain, 1);
                }
            }
        }

        private async Task StartGameAsync()
        {
            try
            {
                await LoadStateAsync();
                socket = new GameSocketClient();
                await socket.ConnectAsync(app.Api.GameSocketUrl(session.session_id));

                countdownText.text = "LOADING";
                countdownText.color = RuntimeUi.White;
                video.Prepare();
                float deadline = Time.realtimeSinceStartup + 20f;
                while (!video.isPrepared && Time.realtimeSinceStartup < deadline) await Task.Delay(40);
                if (!video.isPrepared) throw new Exception("Video master did not prepare in time");

                await ShowCountdownAsync();
                video.time = 0;
                video.SetDirectAudioVolume(0, 1.0f);
                video.Play();
                await socket.SendAsync(new GameActionMessage { action = "start", media_time_ms = 0 });
                started = true;
            }
            catch (Exception e)
            {
                countdownText.text = "CONNECTION ERROR";
                systemText.text = e.Message;
            }
        }

        private async Task ShowCountdownAsync()
        {
            string[] values = { "3", "2", "1", "GO!" };
            foreach (string value in values)
            {
                countdownText.text = value;
                countdownText.color = value == "GO!" ? RuntimeUi.Gold : RuntimeUi.White;
                await Task.Delay(value == "GO!" ? 320 : 610);
            }
            countdownText.text = "";
        }

        private void Update()
        {
            if (socket != null) socket.Drain(HandleMessage);

            float now = Time.unscaledTime;
            foreach (PlayerHud hud in playerHuds)
            {
                if (hud != null && hud.root.activeSelf && hud.gradeUntil > 0 && now > hud.gradeUntil)
                {
                    hud.grade.text = "";
                    hud.gradeUntil = 0;
                }
            }

            if (!started || video == null || !video.isPlaying) return;

            int currentMs = Mathf.Max(0, (int)(video.time * 1000.0));
            int duration = Mathf.Max(dance.duration_ms, 1);
            float p = Mathf.Clamp01(currentMs / (float)duration);
            if (progressFill != null) progressFill.rectTransform.anchorMax = new Vector2(p, 1);
            progressText.text = currentMs / 60000 + ":" + (currentMs / 1000 % 60).ToString("00");
            UpdateCueStrip(currentMs);

            if (Time.unscaledTime - lastClockSend > 0.04f)
            {
                lastClockSend = Time.unscaledTime;
                _ = socket.SendAsync(new MediaClockMessage { media_time_ms = currentMs });
            }
        }

        private void HandleMessage(SocketEnvelope message)
        {
            if (message.type == "score_event")
            {
                PlayerHud hud = FindHud(message);
                if (hud == null) return;

                Color32 playerColor = SlotColor(hud.slot);
                Color32 coachColor = CoachColor(hud.coachIndex);
                if (message.player_pose != null && message.player_pose.Length >= 29)
                    hud.mirror.SetPose(message.player_pose, true, playerColor);

                hud.score.text = message.total_score.ToString("N0");
                hud.combo.text = "COMBO " + message.combo;

                if (message.is_move_grade)
                {
                    hud.grade.text = string.IsNullOrEmpty(message.grade) ? "" : message.grade.ToUpperInvariant();
                    hud.grade.color = message.grade == "x" ? new Color32(255,92,116,255) : coachColor;
                    hud.gradeUntil = Time.unscaledTime + 0.82f;
                }
            }
            else if (message.type == "players_changed")
            {
                if (message.players != null && message.players.Length > 0) ApplyPlayers(message.players);
            }
            else if (message.type == "phone_disconnected")
            {
                systemText.text = "PHONE DISCONNECTED";
                systemText.color = RuntimeUi.Pink;
            }
            else if (message.type == "game_over")
            {
                started = false;
                systemText.text = "ROUTINE COMPLETE";
                systemText.color = RuntimeUi.Gold;
            }
        }

        private void TogglePause()
        {
            if (!started || video == null) return;
            paused = !paused;
            if (paused)
            {
                video.Pause();
                systemText.text = "PAUSED";
                _ = socket.SendAsync(new GameActionMessage { action = "pause" });
            }
            else
            {
                systemText.text = "";
                video.Play();
                _ = socket.SendAsync(new GameActionMessage { action = "resume" });
            }
        }

        private void OnVideoFinished(VideoPlayer source)
        {
            if (socket != null) _ = socket.SendAsync(new MediaClockMessage { media_time_ms = dance.duration_ms });
        }

        private void Exit() { app.ShowLibrary(); }

        private void OnDestroy()
        {
            if (app != null)
            {
                app.Input.Cancel -= Exit;
                app.Input.Submit -= TogglePause;
            }
            if (socket != null) socket.Dispose();
            if (video != null) video.Stop();
            if (videoTexture != null) videoTexture.Release();
        }
    }

    public sealed class CoachStage3D : MonoBehaviour
    {
        private DanceApiClient api; private string danceId; private VideoPlayer clock; private RawImage sourceVideo; private GameObject coach; private Camera coachCamera;
        private RenderTexture coachTexture; private HumanoidPoseDriver driver; private PoseTimeline timeline; private bool active3D;

        public void Initialize(Canvas canvas, DanceApiClient client, string id, VideoPlayer videoClock, RawImage videoImage)
        {
            if (!AppConfig.Enable3DCoach) return;
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
