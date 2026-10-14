#include <jni.h>
#include <GLES2/gl2.h>
#include <android/log.h>
extern "C" JNIEXPORT jint JNICALL Java_com_tsto_glprobe_MainActivity_nativeCreateShader(JNIEnv*,jclass){
 __android_log_print(ANDROID_LOG_INFO,"TSTO_GLPROBE","JNI arm64 before glCreateShader");
 GLuint id=glCreateShader(GL_VERTEX_SHADER);
 __android_log_print(ANDROID_LOG_INFO,"TSTO_GLPROBE","JNI arm64 glCreateShader returned %u",id);
 return static_cast<jint>(id);
}
