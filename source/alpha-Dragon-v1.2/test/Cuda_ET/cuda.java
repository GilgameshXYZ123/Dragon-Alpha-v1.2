/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class cuda 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp3(alpha.MEM_1GB * 6);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static void test1() throws IOException{
//        System.out.println(Cuda.nvidia_smi());
//
        Cuda.nvidia_smi_view("", 1, false);
    }
    
    
    public static void test2() {
        Tensor X = eg.empty(512, 128, 128, 32); X.delete();
        
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<10000; i++) {
            X = eg.empty(512, 128, 128, 32); X.delete();
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void test3() {
        float a = -1.5957692f;
        float b =  0.044715f;
        System.out.println(2*a*b);
    }
    
    public static void test4() {
        float x = -10.250525f;
        float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
        float expuf = (float) Math.exp(-u);
        double expu = Math.exp(u);
        System.out.println(u + ", " + expu + ", " + expuf);
        
        float k = expuf /(1 + expuf);
        System.out.println(k);
        
    }
    
    public static void test5(){
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<10000; i++) {
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventSync_Del(event);
//            Cuda.eventSynchronize(event);
//            Cuda.deleteEvent(event);
        }
        System.out.println(timer.record().timeStamp_dif_millis());
    }
    
    
    public static void main(String[] args) 
    {
        try
        {
            test1();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
    }
}
