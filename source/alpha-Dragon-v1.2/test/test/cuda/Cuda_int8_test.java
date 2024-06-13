/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_int8_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void testCorrect(int IH, int IW) {
        int length = IH * IW;
        byte[] a = Vector.random_byte_vector(length);
        Tensor X = eg.tensor_int8(a, IH, IW);
        byte[] b = X.value_int8();
        
        float sp = Vector.samePercent_absolute(a, b);
        Vector.println(a, 0, 10);
        Vector.println(b, 0, 10);
        System.out.println("sp = " + sp);
        
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testSpeed(int IH, int IW) {
        int length = IH * IW;
        byte[] a = Vector.random_byte_vector(length);
        Tensor X = eg.tensor_int8(a, IH, IW);
        
        int nIter = 100;
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<nIter; i++)
        {
            //(1024*1024, 3): 2.639358 GB/s
            //(1024, 1024): 5.744485 GB/s
//            X = eg.tensor_int8(a, IH, IW); X.delete();
            
            //(1024*1024*3): 1.600922 GB/s
            //(1024, 1024): 3.051758 GB/s
            byte[] b = X.value_int8();
        }
        timer.record();
        
        float time = (float) timer.timeStamp_dif_millis() / nIter;
	int data_size = length;
        
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
    }
    
    public static void main(String[] args)
    {
//        for(int ih=1; ih<=128; ih++)
//        for(int iw=1; iw<=128; iw++)
//            testCorrect(ih, iw);
        
//        testSpeed(1024, 1024);
        testCorrect(1024*1024, 3);
        testSpeed(1024*1024, 3);
        
        //testSpeed(512*32*32, 3);
    }
}
