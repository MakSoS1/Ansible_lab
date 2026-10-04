using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DanceFlow.UnityClient.Editor
{
    [InitializeOnLoad]
    public static class ProjectBootstrap
    {
        private const string ScenePath = "Assets/Scenes/DanceFlowMain.unity";
        static ProjectBootstrap() { EditorApplication.delayCall += EnsureProject; }

        [MenuItem("DanceFlow/Bootstrap Unity Client")]
        public static void EnsureProject()
        {
            Directory.CreateDirectory("Assets/Scenes");
            Directory.CreateDirectory("Assets/Resources/Coaches");
            Directory.CreateDirectory("Assets/Resources/Backgrounds");

            PlayerSettings.companyName = "DanceFlow";
            PlayerSettings.productName = "DanceFlow";
            PlayerSettings.defaultScreenWidth = 3840;
            PlayerSettings.defaultScreenHeight = 2160;
            PlayerSettings.fullScreenMode = FullScreenMode.FullScreenWindow;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            TryEnableInputSystem();

            if (!File.Exists(ScenePath))
            {
                Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                GameObject app = new GameObject("DanceFlowApp");
                app.AddComponent<DanceFlowApp>();
                EditorSceneManager.SaveScene(scene, ScenePath);
            }

            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("DanceFlow Unity client bootstrapped. Put a Humanoid prefab at Assets/Resources/Coaches/DefaultCoach.prefab to enable 3D coach mode.");
        }

        private static void TryEnableInputSystem()
        {
            Object[] assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/ProjectSettings.asset");
            if (assets == null || assets.Length == 0) return;
            SerializedObject settings = new SerializedObject(assets[0]);
            SerializedProperty input = settings.FindProperty("activeInputHandler") ?? settings.FindProperty("m_ActiveInputHandler");
            if (input != null) { input.intValue = 1; settings.ApplyModifiedPropertiesWithoutUndo(); }
        }
    }
}
