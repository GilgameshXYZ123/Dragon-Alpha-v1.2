/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Syncer;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Sync_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024)
            .sync(false);
    
    public static void test1() {
        Tensor X = eg.Gaussian(128, 224, 224, 3);
        
        Syncer sc = X.syncer();
        Thread t = new Thread(()-> { 
            while(true) {
                X.syncer().sync(); 
            }
        });
        t.start();
        
        sc.sync(); 
    }
    
    
    public static void main(String[] args)
    {
        
        test1();
    }
}