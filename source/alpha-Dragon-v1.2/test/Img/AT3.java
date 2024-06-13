/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import static Img.AT2.path;
import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.common.DragonCV.cv;

/**
 *
 * @author Gilgamesh
 */
public class AT3 
{
    public static byte pix_restrict(float v) {
        if(v < 0) return 0;
        if(v > 255) return (byte) 255;
        return (byte) v;
    }
    
    public static byte[] saturation(byte[] pixel, int IH, int IW, float value) {
        byte[] pixel2 = new byte[IH * IW * 3];
        for (int i = 0; i < pixel.length; i += 3) {
            int r = pixel[i    ] & 0xff; 
            int g = pixel[i + 1] & 0xff;
            int b = pixel[i + 2] & 0xff;
            
            float gray = 0.2989f*r + 0.5870f*g + 0.1140f*b; //weights from CCIR 601 spec
            float fr = -gray * value + r * (1+value);
            float fg = -gray * value + g * (1+value);
            float fb = -gray * value + b * (1+value);
        
            pixel2[i  ] = pix_restrict(fr);
            pixel2[i+1] = pix_restrict(fg);
            pixel2[i+2] = pix_restrict(fb);
        }
        return pixel2;
    }
    
    public static byte[] saturation2(byte[] pixel, int IH, int IW, float value) {
        byte[] pixel2 = new byte[IH * IW * 3];
        for (int i = 0; i < pixel.length; i += 3) {
            int r = pixel[i    ] & 0xff; 
            int g = pixel[i + 1] & 0xff;
            int b = pixel[i + 2] & 0xff;
            
            float gray = (r + g + b) / 3.0f; //weights from CCIR 601 spec
            float fr = -gray * value + r * (1+value);
            float fg = -gray * value + g * (1+value);
            float fb = -gray * value + b * (1+value);
        
            pixel2[i  ] = pix_restrict(fr);
            pixel2[i+1] = pix_restrict(fg);
            pixel2[i+2] = pix_restrict(fb);
        }
        return pixel2;
    }
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img, "original");
        int IH = img.getHeight();
        int IW = img.getWidth();
        byte[] pixel = cv.pixel(img);
        
        byte[] pixel1 = saturation(pixel, IH, IW, 0.5f);
        img = cv.BGR(pixel1, IH,IW); cv.imshow(img, "2");
        System.out.println(cv.brief(img));
        
        byte[] pixel2 = saturation2(pixel, IH, IW, 0.5f);
        img = cv.BGR(pixel2, IH,IW); cv.imshow(img, "3");
        System.out.println(cv.brief(img));
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
