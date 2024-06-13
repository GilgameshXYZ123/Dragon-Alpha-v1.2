/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_init 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static ExRandom exr = new ExRandom();
    
    public static void main(String[] args) 
    {
        try {
            Cuda.getDeviceId();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
    }
}
