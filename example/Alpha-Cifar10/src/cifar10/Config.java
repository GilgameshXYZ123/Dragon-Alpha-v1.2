/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Config 
{
    public static String alpha_home = "C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2";
   
    static { alpha.home(Config.alpha_home); }
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    public static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
}
