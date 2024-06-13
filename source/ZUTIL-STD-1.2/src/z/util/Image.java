/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/**
 *
 * @author Gilgamesh
 */
public class Image 
{
    public static int getChannelNum(BufferedImage img)
    {
        return img.getAlphaRaster()!=null? 4: 3;
    }
    
    public static byte[] getPixels(BufferedImage img)
    {
        return ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
    }
    
    public static float[] getFPixelsOrderByChannel(BufferedImage img)
    {
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        float[] fpixels = new float[pixels.length];
        
        int index = 0;
        for(int i=0; i<pixels.length; i+=3) fpixels[index++] = pixels[i];
        for(int i=1; i<pixels.length; i+=3) fpixels[index++] = pixels[i];
        for(int i=2; i<pixels.length; i+=3) fpixels[index++] = pixels[i];
        return fpixels;
    }
}
