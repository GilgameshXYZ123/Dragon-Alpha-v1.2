/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Raw;

import static Raw.raw4.eg;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class HSI_present1
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    public static int[] read_shape(String fname) {
        int[] shape = new int[3];
        alpha.fl.for_each_line(fname + ".hdr", (String line)->{
            String substr = null;
            if (line.startsWith("samples = ")) shape[0] = Integer.valueOf(line.substring(line.lastIndexOf('=') + 2));
            else if(line.startsWith("bands = ")) shape[2] = Integer.valueOf(line.substring(line.lastIndexOf('=') + 2));
            else if(line.startsWith("lines = ")) shape[1] = Integer.valueOf(line.substring(line.lastIndexOf('=') + 2));
        });
        return shape;
    } 
    
    public static void test() throws IOException  {
        Tensor raw = eg.img.read_raw_bil_dtype12(IMG + ".raw", read_shape(IMG));
        Tensor white = eg.img.read_raw_bil_dtype12(WHITE + ".raw", read_shape(WHITE));
        Tensor dark = eg.img.read_raw_bil_dtype12(DARK + ".raw", read_shape(DARK));
        
        cv.imshow(eg.img.BGR(raw), "raw(BGR type)");
//        cv.imshow(eg.img.BGR(white), "white(BGR type)");
//        cv.imshow(eg.img.BGR(dark), "dark(BGR type)");
        
        cv.imshow(eg.img.gray(raw), "raw(Gray type)");
//        cv.imshow(eg.img.gray(white), "white(Gray type)");
//        cv.imshow(eg.img.gray(dark), "dark(Gray type)");
        
        white = eg.img.resize(true, white, raw.dim(0), raw.dim(1));
        dark = eg.img.resize(true, dark, raw.dim(0), raw.dim(1));
        
        Tensor raw2 = eg.img.reflection_normalization(true, raw, white, dark)
                .img().linear_dtype_to_pixel(true, 255, 0);
        
        cv.imshow(eg.img.BGR(raw2), "raw2(BGR type)");
        cv.imshow(eg.img.gray(raw2), "raw2(Gray type)");
        
        System.gc();
    }

    static String dir = "H:\\virtual-disc-V-dataset\\[p] FX17-corn-2023-12-24\\RD-anhui-mengcheng-shangdu188\\anhui-mengcheng-shangdu188_emptyname_0000\\capture\\";
    static String IMG   = dir + "anhui-mengcheng-shangdu188_emptyname_0000";
    static String WHITE = dir + "WHITEREF_anhui-mengcheng-shangdu188_emptyname_0000";
    static String DARK  = dir + "DARKREF_anhui-mengcheng-shangdu188_emptyname_0000";
    
    public static void main(String[] args) {
        try
        {
            double v =   1.0 * 32 / 80405325;
            System.out.format("%Ef", v);
//            test();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
   
}
