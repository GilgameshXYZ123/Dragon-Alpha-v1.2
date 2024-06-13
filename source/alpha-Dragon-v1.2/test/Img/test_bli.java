package Img;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



/**
 *
 * @author Gilgamesh
 */
public class test_bli 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    public static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static void main(String[] args)
    {
        eg.img.read_raw_bil_dtype12("H:\\virtual-disc-V-dataset\\[p] pepper-2024-1-6\\RD-cuiluo3_qin-16\\"
                + "cuiluo3_qin-16_emptyname_0000\\capture\\"
                + "cuiluo3_qin-16_emptyname_0000.raw", 584, 640, 224);
    }
    
}
