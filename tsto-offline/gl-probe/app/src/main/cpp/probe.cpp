#include <jni.h>
#include <GLES2/gl2.h>
#include <android/log.h>
extern "C" JNIEXPORT jint JNICALL Java_com_tsto_glprobe_MainActivity_nativeCreateShader(JNIEnv*,jclass){
 __android_log_print(ANDROID_LOG_INFO,"TSTO_GLPROBE","JNI arm64 before glCreateShader");
 GLuint id=glCreateShader(GL_VERTEX_SHADER);
 __android_log_print(ANDROID_LOG_INFO,"TSTO_GLPROBE","JNI arm64 glCreateShader returned %u",id);
 return static_cast<jint>(id);
}

extern "C" JNIEXPORT jint JNICALL Java_com_tsto_glprobe_MainActivity_nativeStressShaders(JNIEnv*,jclass,jint count){
  if(count<1 || count>1024) return -1;
  GLuint shaders[1024]={};
  for(int i=0;i<count;++i){
    shaders[i]=glCreateShader(GL_VERTEX_SHADER);
    if(shaders[i]==0){
      __android_log_print(ANDROID_LOG_ERROR,"TSTO_GLPROBE","FAIL shader stress at %d / %d",i,count);
      for(int j=0;j<i;++j)glDeleteShader(shaders[j]);
      return -i-1;
    }
  }
  for(int i=0;i<count;++i)glDeleteShader(shaders[i]);
  return count;
}
