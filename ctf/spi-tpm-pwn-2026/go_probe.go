package main

import (
    "fmt"
    "os"
    "os/signal"
    "runtime"
    "sync"
    "sync/atomic"
    "syscall"
    "time"
    "unsafe"
)

type genRequest struct {
    Key uint32
    Status int32
    Reserved [2]uint64
}
type cryptRequest struct {
    Key uint32
    Cost uint32
    InputLen uint32
    OutputLen uint32
    Capacity uint32
    Status int32
    Input uint64
    Output uint64
    Reserved [2]uint64
}
const genIoctl uintptr = 0xc0184301
const encIoctl uintptr = 0xc0384302

func openDevice() (int, error) {
    return syscall.Open("/dev/spi-tpm0", syscall.O_RDWR|syscall.O_CLOEXEC, 0)
}
func ioctl(fd int, request uintptr, arg unsafe.Pointer) syscall.Errno {
    _, _, err := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), request, uintptr(arg))
    return err
}
func generateKey() (uint32, error) {
    fd, err := openDevice()
    if err != nil { return 0, err }
    defer syscall.Close(fd)
    var req genRequest
    er := ioctl(fd,genIoctl,unsafe.Pointer(&req))
    if er != 0 || req.Status != 0 || req.Key == 0 {
        return 0,fmt.Errorf("key ioctl errno=%d status=%d key=%d",er,req.Status,req.Key)
    }
    runtime.KeepAlive(req)
    return req.Key,nil
}
func worker(wg *sync.WaitGroup, workerID int, key uint32, attempts int, anomalies *atomic.Int64) {
    defer wg.Done()
    fd,err:=openDevice()
    if err != nil {
        anomalies.Add(1)
        fmt.Printf("WORKER%d open_error=%v\n",workerID,err)
        return
    }
    defer syscall.Close(fd)
    runtime.LockOSThread()
    defer runtime.UnlockOSThread()
    in:=make([]byte,4000)
    out:=make([]byte,4096)
    for i:=range in { in[i]=byte(0x41 +workerID) }
    start:=time.Now()
    for i:=0;i<attempts;i++ {
        n:=4000
        if i%2 == 1 { n=8 }
        req:=cryptRequest{
            Key:key,Cost:1,InputLen:uint32(n),Capacity:uint32(len(out)),
            Input:uint64(uintptr(unsafe.Pointer(&in[0]))),
            Output:uint64(uintptr(unsafe.Pointer(&out[0]))),
        }
        t:=time.Now()
        er:=ioctl(fd,encIoctl,unsafe.Pointer(&req))
        dt:=time.Since(t)
        expected:=uint32(((n+3)&^3)+40)
        invalid:=er!=0 || req.Status!=0 || req.OutputLen!=expected ||
          len(out)<4 || string(out[:4])!="STP1"
        runtime.KeepAlive(in)
        runtime.KeepAlive(out)
        runtime.KeepAlive(req)
        if invalid {
            anomalies.Add(1)
            fmt.Printf("WORKER%d FAIL i=%d errno=%d status=%d output=%d expected=%d dt=%s\n",workerID,i,er,req.Status,req.OutputLen,expected,dt)
        } else if i==0 || i%40 ==0 {
            fmt.Printf("WORKER%d PASS i=%d len=%d dt=%s\n",workerID,i,n,dt)
        }
    }
    fmt.Printf("WORKER%d DONE elapsed=%s\n",workerID,time.Since(start))
}
func main() {
    runtime.GOMAXPROCS(2)
    fmt.Printf("PROBE_START uid=%d gid=%d arch=%s cpus=%d\n",os.Getuid(),os.Getgid(),runtime.GOARCH,runtime.NumCPU())
    if unsafe.Sizeof(genRequest{})!=24 || unsafe.Sizeof(cryptRequest{})!=56 {
        fmt.Println("ABI_SIZE_MISMATCH")
        os.Exit(2)
    }
    key,err:=generateKey()
    if err!=nil {fmt.Println("KEY_ERR",err);os.Exit(2)}
    fmt.Printf("KEY_ID=%d\n",key)
    ch:=make(chan os.Signal,64)
    signal.Notify(ch,syscall.SIGUSR1)
    defer signal.Stop(ch)
    var wg sync.WaitGroup
    var failures atomic.Int64
    finished:=make(chan struct{})
    go func() {
        tick:=time.NewTicker(7*time.Millisecond)
        defer tick.Stop()
        for {
            select {
            case <-finished:return
            case <-tick.C:syscall.Kill(os.Getpid(),syscall.SIGUSR1)
            }
        }
    }()
    go func(){
        for {
            select {
            case <-finished:return
            case <-ch:
            }
        }
    }()
    wg.Add(2)
    start:=time.Now()
    go worker(&wg,0,key,160,&failures)
    go worker(&wg,1,key,160,&failures)
    wg.Wait()
    close(finished)
    fmt.Printf("PROBE_DONE failures=%d elapsed=%s\n",failures.Load(),time.Since(start))
}
