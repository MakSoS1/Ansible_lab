/* Local fake-flag diagnostic: freestanding raw_probe helpers included at build time. */
static u8 flood[4096];
int main(void){
 say("PRESSURE_START\n");
 int fd=call(56,-100,(long)"/dev/spi-tpm0",2,0,0,0);struct gen g;fill(&g,0,sizeof(g));long r=call(29,fd,0xc0184301UL,(long)&g,0,0,0);logrow(-1,-1,r,g.status,g.key,0);if(r||g.status)return 1;
 volatile u32 *shared=(void*)call(222,0,4096,3,0x21,-1,0);if((long)shared<0)return 2;*shared=0;
 long child=call(220,17,0,0,0,0,0);
 if(!child){u64 mask=2;call(122,0,8,(long)&mask,0,0,0);fill(flood,'Z',sizeof(flood));long t[2]={0,10000000};call(101,(long)t,0,0,0,0,0);for(int i=0;i<4096 && !*shared;i++)call(64,1,(long)flood,sizeof(flood),0,0,0);call(94,0,0,0,0,0,0);}
 u64 mask=1;call(122,0,8,(long)&mask,0,0,0);fill(input,0x42,sizeof(input));
 say("BACKPRESSURE_READY\n");
 for(int i=0;i<48;i++){
  struct crypt c;fill(&c,0,sizeof(c));c.key=g.key;c.cost=1;c.inlen=8;c.cap=4096;c.in=(u64)input;c.out=(u64)output;
  long start=now();r=call(29,fd,0xc0384302UL,(long)&c,0,0,0);long dt=now()-start;
  logrow(0,i,r,c.status,c.outlen,dt);
 }
 *shared=1;int st;call(260,child,(long)&st,0,0,0,0);say("PROBE_COMPLETE\n");return 0;
}
__attribute__((naked)) void _start(void){__asm__("mov x29,xzr\nmov x30,xzr\nbl main\nmov x8,#94\nsvc #0\nb .");}
