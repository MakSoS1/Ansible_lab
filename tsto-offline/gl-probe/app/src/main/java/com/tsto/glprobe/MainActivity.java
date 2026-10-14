package com.tsto.glprobe;
import android.app.Activity;
import android.opengl.GLSurfaceView;
import android.opengl.GLES20;
import android.os.Bundle;
import android.util.Log;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
public class MainActivity extends Activity {
 private static final String TAG="TSTO_GLPROBE";
 private GLSurfaceView view;
 static {System.loadLibrary("glprobe");}
 private static native int nativeCreateShader();
 private static native int nativeStressShaders(int count);
 @Override protected void onCreate(Bundle state){
  super.onCreate(state);view=new GLSurfaceView(this);view.setEGLContextClientVersion(2);
  view.setRenderer(new GLSurfaceView.Renderer(){
   @Override public void onSurfaceCreated(GL10 unused,EGLConfig cfg){
    Log.i(TAG,"ENTER native glCreateShader in ARM64 app");
    int shader=nativeCreateShader();
    if(shader<=0){Log.e(TAG,"FAIL ARM64 native shader id is zero");return;}
    Log.i(TAG,"PASS native ARM64 glCreateShader="+shader);
    int js=GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
    if(js<=0){Log.e(TAG,"FAIL Java shader id is zero");return;}
    Log.i(TAG,"PASS Java GLES20 glCreateShader="+js);
    GLES20.glDeleteShader(shader);GLES20.glDeleteShader(js);
    int stress=nativeStressShaders(256);
    if(stress!=256){Log.e(TAG,"FAIL native ARM64 shader stress count="+stress);return;}
    Log.i(TAG,"PASS native ARM64 shader stress="+stress);
   }
   @Override public void onSurfaceChanged(GL10 gl,int w,int h){GLES20.glViewport(0,0,w,h);}
   @Override public void onDrawFrame(GL10 gl){GLES20.glClearColor(0,0,0,1);GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);}
  });setContentView(view);
 }
 @Override protected void onPause(){super.onPause();view.onPause();}
 @Override protected void onResume(){super.onResume();view.onResume();}
}
