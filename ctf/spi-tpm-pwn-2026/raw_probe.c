/* Bounded CTF guest diagnostic. Compile for AArch64; never execute on the host. */
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned char u8;
static long call(long n,long a,long b,long c,long d,long e,long f){
 register long x8 __asm__("x8")=n; register long x0 __asm__("x0")=a;
 register long x1 __asm__("x1")=b; register long x2 __asm__("x2")=c;
 register long x3 __asm__("x3")=d; register long x4 __asm__("x4")=e;
 register long x5 __asm__("x5")=f;
 __asm__ volatile("svc 0":"+r"(x0):"r"(x8),"r"(x1),"r"(x2),"r"(x3),"r"(x4),"r"(x5):"memory"); return x0;
}
static void fill(void *p,u8 v,u64 n){u8 *b=p;while(n--)*b++=v;}
static int slen(const char *p){int i=0;while(p[i])i++;return i;}
static void say(const char *s){call(64,1,(long)s,slen(s),0,0,0);}
static long now(void){long t[2];call(113,1,(long)t,0,0,0,0);return t[0]*1000000+t[1]/1000;}
static char line[256];static int lp;
static void str(const char *p){while(*p)line[lp++]=*p++;}
static void num(long n){char a[24];int i=0;if(n<0){line[lp++]='-';n=-n;}do{a[i++]='0'+n%10;n/=10;}while(n);while(i)line[lp++]=a[--i];}
static void logrow(long cpu,long i,long r,long st,long size,long dt){lp=0;str("CPU=");num(cpu);str(" i=");num(i);str(" rc=");num(r);str(" status=");num(st);str(" size=");num(size);str(" us=");num(dt);line[lp++]='\n';call(64,1,(long)line,lp,0,0,0);}
struct crypt {u32 key,cost,inlen,outlen,cap;int status;u64 in,out,res[2];};
struct gen {u32 key;int status;u64 res[2];};
static u8 input[4056],output[4096];
static void worker(int cpu,u32 key){
 u64 mask=1UL<<cpu;call(122,0,8,(long)&mask,0,0,0);
 int fd=call(56,-100,(long)"/dev/spi-tpm0",2,0,0,0);fill(input,0x41+cpu,sizeof(input));
 for(int i=0;i<128;i++){
  struct crypt c;fill(&c,0,sizeof(c));c.key=key;c.cost=1;c.inlen=(i&1)?8:4056;c.cap=4096;c.in=(u64)input;c.out=(u64)output;
  long start=now();long r=call(29,fd,0xc0384302UL,(long)&c,0,0,0);long dt=now()-start;
  if(i<4||i%32==0||r||c.status||dt<60000||c.outlen!=c.inlen+40)logrow(cpu,i,r,c.status,c.outlen,dt);
 }
 call(57,fd,0,0,0,0,0);say("WORKER_COMPLETE\n");
}
int main(void){
 lp=0;str("RAW_PROBE uid=");num(call(174,0,0,0,0,0,0));str(" pid=");num(call(172,0,0,0,0,0,0));str("\n");call(64,1,(long)line,lp,0,0,0);
 int fd=call(56,-100,(long)"/dev/spi-tpm0",2,0,0,0);struct gen g;fill(&g,0,sizeof(g));long r=call(29,fd,0xc0184301UL,(long)&g,0,0,0);call(57,fd,0,0,0,0,0);
 logrow(-1,-1,r,g.status,g.key,0);if(r||g.status)return 1;
 long pid=call(220,17,0,0,0,0,0);if(pid<0){say("fork unavailable\n");worker(0,g.key);}else if(!pid){worker(1,g.key);call(94,0,0,0,0,0,0);}else{worker(0,g.key);int status;call(260,pid,(long)&status,0,0,0,0);}
 say("PROBE_COMPLETE\n");return 0;
}
__attribute__((naked)) void _start(void){__asm__("mov x29,xzr\nmov x30,xzr\nbl main\nmov x8,#94\nsvc #0\nb .");}
