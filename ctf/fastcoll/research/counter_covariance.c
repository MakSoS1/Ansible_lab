/* Offline research only: this is NOT a live FastColl solver.
 * Build: cc -O3 -Wall -Wextra counter_covariance.c -Wl,-l:libxxhash.so.0 -o covariance
 * Run: ./covariance 0x123456789abcdef0 1000000 0 16
 * Args: true seed offset, max candidates, construction 0..3, stripe count 4..16.
 * Mode 0 targets alternating finalizer keys; mode 1 low keys; mode 2 high
 * keys; mode 3 uses the derived guessed secret directly as input stripes.
 * Only seed i (not the hidden offset) is given to the message generator.
 * No network, archive code, environment credentials or privileged operations.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
typedef struct { uint64_t low64, high64; } XXH128_hash_t;
extern XXH128_hash_t XXH3_128bits_withSeed(const void*, size_t, uint64_t);
static const uint8_t DS[192]={
0xb8,0xfe,0x6c,0x39,0x23,0xa4,0x4b,0xbe,0x7c,0x01,0x81,0x2c,0xf7,0x21,0xad,0x1c,
0xde,0xd4,0x6d,0xe9,0x83,0x90,0x97,0xdb,0x72,0x40,0xa4,0xa4,0xb7,0xb3,0x67,0x1f,
0xcb,0x79,0xe6,0x4e,0xcc,0xc0,0xe5,0x78,0x82,0x5a,0xd0,0x7d,0xcc,0xff,0x72,0x21,
0xb8,0x08,0x46,0x74,0xf7,0x43,0x24,0x8e,0xe0,0x35,0x90,0xe6,0x81,0x3a,0x26,0x4c,
0x3c,0x28,0x52,0xbb,0x91,0xc3,0x00,0xcb,0x88,0xd0,0x65,0x8b,0x1b,0x53,0x2e,0xa3,
0x71,0x64,0x48,0x97,0xa2,0x0d,0xf9,0x4e,0x38,0x19,0xef,0x46,0xa9,0xde,0xac,0xd8,
0xa8,0xfa,0x76,0x3f,0xe3,0x9c,0x34,0x3f,0xf9,0xdc,0xbb,0xc7,0xc7,0x0b,0x4f,0x1d,
0x8a,0x51,0xe0,0x4b,0xcd,0xb4,0x59,0x31,0xc8,0x9f,0x7e,0xc9,0xd9,0x78,0x73,0x64,
0xea,0xc5,0xac,0x83,0x34,0xd3,0xeb,0xc3,0xc5,0x81,0xa0,0xff,0xfa,0x13,0x63,0xeb,
0x17,0x0d,0xdd,0x51,0xb7,0xf0,0xda,0x49,0xd3,0x16,0x55,0x26,0x29,0xd4,0x68,0x9e,
0x2b,0x16,0xbe,0x58,0x7d,0x47,0xa1,0xfc,0x8f,0xf8,0xb8,0xd1,0x7a,0xd0,0x31,0xce,
0x45,0xcb,0x3a,0x8f,0x95,0x16,0x04,0x28,0xaf,0xd7,0xfb,0xca,0xbb,0x4b,0x40,0x7e};
static const uint64_t INIT[8]={0xc2b2ae3d,0x9e3779b185ebca87ULL,0xc2b2ae3d27d4eb4fULL,0x165667b19e3779f9ULL,0x85ebca77c2b2ae63ULL,0x85ebca77,0x27d4eb2f165667c5ULL,0x9e3779b1};
static uint64_t u64(const void *p){uint64_t x;memcpy(&x,p,8);return x;}
static void p64(void *p,uint64_t x){memcpy(p,&x,8);}
static void secret(uint8_t *out,uint64_t seed){for(int i=0;i<24;i++) p64(out+8*i,u64(DS+8*i)+(i&1?-seed:seed));}
static void accumulate(uint64_t *acc,const uint8_t *m, const uint8_t *sec,int n){
memcpy(acc,INIT,sizeof(INIT));
for(int s=0;s<n;s++){int offset=s==n-1?121:8*s;
for(int l=0;l<8;l++){uint64_t d=u64(m+64*s+8*l), x=d^u64(sec+offset+8*l);acc[l]+=(uint64_t)(uint32_t)x*(x>>32);acc[l^1]+=d;}}}
static void forge(uint8_t *m,uint64_t seed,int mode,int n){
uint8_t sec[192];uint64_t acc[8],targets[8];secret(sec,seed);
for(int s=0;s<n;s++)for(int l=0;l<8;l++)p64(m+64*s+8*l,u64(sec+(s==n-1?121:8*s)+8*l));
if(mode==3)return;
accumulate(acc,m,sec,n);
for(int l=0;l<8;l++){
 int off=mode==1?11:(mode==2?117:(l&1?117:11)); targets[l]=u64(sec+off+8*l);
 int other=l^1;uint64_t k0=u64(sec+8*other),k1=u64(sec+8+8*other),delta=targets[l]-acc[l];
 uint32_t newlow=(uint32_t)k1+(uint32_t)delta;
 uint64_t lowdelta=(uint64_t)newlow-(uint32_t)k1;uint64_t remaining=delta-lowdelta;
 uint32_t newhigh=(uint32_t)(k0>>32)+(uint32_t)(remaining>>32);
 p64(m+8*other,((uint64_t)newhigh<<32)|(uint32_t)k0);
 p64(m+64+8*other,(k1&0xffffffff00000000ULL)|newlow);
}}
struct Item{uint64_t lo,hi;uint32_t i,used;};
int main(int argc,char **argv){
 uint64_t C=argc>1?strtoull(argv[1],0,0):0x123456789abcdef0ULL;
 unsigned count=argc>2?strtoul(argv[2],0,0):1000000;
 int mode=argc>3?atoi(argv[3]):0,n=argc>4?atoi(argv[4]):16;
 if(argc>5 || count<1 || count>1000000 || mode<0 || mode>3 || n<4 || n>16){
 fprintf(stderr,"usage: covariance [offset [count<=1000000 [mode=0..3 [stripes=4..16]]]]\n");return 2;}
 uint8_t msg[1024];forge(msg,0,0,16);XXH128_hash_t h=XXH3_128bits_withSeed(msg,1024,0);
 if(h.high64!=0xdbfd52f5f5a41471ULL || h.low64!=0xdbbb3ad8d9bd8268ULL){fprintf(stderr,"bad forge self test\n");return 2;}
 unsigned size=1;while(size<2*count)size<<=1;
 struct Item *tab=calloc(size,sizeof(*tab));if(!tab)return 3;unsigned col=0,processed=0;
 for(unsigned i=0;i<count;i++){
  processed++;forge(msg,i,mode,n);h=XXH3_128bits_withSeed(msg,64*n,C+i);
  unsigned p=h.low64&(size-1);
  while(tab[p].used){
   if(tab[p].lo==h.low64 && tab[p].hi==h.high64){
    uint8_t previous[1024];forge(previous,tab[p].i,mode,n);
    if(memcmp(msg,previous,64*n)==0)break;
    printf("COLLISION C=%016" PRIx64 " i=%u j=%u digest=%016" PRIx64 "%016" PRIx64 "\n",C,tab[p].i,i,h.high64,h.low64);col++;break;}
   p=(p+1)&(size-1);
  }
  if(!tab[p].used)tab[p]=(struct Item){h.low64,h.high64,i,1};
  if(col>=10)break;
 }
 printf("done mode=%d n=%d requested=%u processed=%u collisions=%u\n",mode,n,count,processed,col);free(tab);return 0;
}
