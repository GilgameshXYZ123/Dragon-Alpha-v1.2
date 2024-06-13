/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Demo 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static int pixel_to_int(byte b) {
        if(b >= 0) return b;
        else return -b + 127;
    }
    
    public static byte int_to_pixel(int x) {
        if(x < 0) x = 0; else if(x > 255) x = 255;
        return (byte) x;
    }
    
    public static void adjust_bright(byte[] pixel, int height, int width, int up) {
        int length = height * width;
        for(int i=0; i<length; i ++) {
            int bidx = i*3, ridx = bidx + 1, gidx = bidx + 2;
            int blue  = pixel[bidx] & 0xff;
            int red   = pixel[ridx] & 0xff;
            int green = pixel[gidx] & 0xff;
            
            //process-----------------------------------------------------------
            blue += up;
            red += up;
            green += up;
            //process-----------------------------------------------------------
            
            pixel[bidx] = int_to_pixel(blue);//blue
            pixel[gidx] = int_to_pixel(red);//green
            pixel[ridx] = int_to_pixel(green);//red
        }
    } 
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img, "original");
        int height = img.getHeight();
        int width = img.getWidth();
        byte[] pixel = cv.pixel(img);
        System.out.println(cv.brief(img));

        //process pixel---------------------------------------------------------
        adjust_bright(pixel, height, width, 115);
        
        //process pixel---------------------------------------------------------
        BufferedImage img2 = cv.BGR(pixel, height, width); cv.imshow(img2);
        System.out.println(cv.brief(img2));
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
