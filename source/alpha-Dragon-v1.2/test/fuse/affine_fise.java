/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fuse;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.simple.affine.Affine;

/**
 *
 * @author Gilgamesh
 */
public class affine_fise 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp3(alpha.MEM_1GB * 6);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static void test1()
    {
        double a = -2 * Math.sqrt(2.0 / Math.PI);
        double b = 0.044715;
        double c = 2*a*b;
        
        System.out.println((float)a);
        System.out.println((float)c);
    }
    
    public static void main(String[] args)
    {
        test1();
    }
}
