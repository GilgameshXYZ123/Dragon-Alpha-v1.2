/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

/**
 *
 * @author Gilgamesh
 */
public class CLIP 
{
    public static void main(String[] args)
    {
//        float x = 1.0f;
//        float max = Float.MAX_VALUE;
//        
//        System.out.println(0 * max);
//        System.out.println(max - x == max);
        
        int GK = 128 * 96 * 96;
        int block = 6;
        int GZ = 16;
        
        double x = Math.sqrt(GK / 128) * block * GZ / 39;
        System.out.println((int)x);
        
        
    }
}
