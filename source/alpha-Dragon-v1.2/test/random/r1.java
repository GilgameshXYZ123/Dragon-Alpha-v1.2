/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package random;

import java.util.Random;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class r1 
{
    private static final long multiplier = 0x5DEECE66DL;
    private static final long addend = 0xBL;
    private static final long mask = (1L << 48) - 1;
    
    
    public static void test1()
    {
        System.out.println(0x5DEECE66DL);
        System.out.println((1 << 31) - 1);
    }
    
    public static void test2() {
        ExRandom exr = new ExRandom();
        exr.setSeed(100000);
        for(int i=0; i<10; i++)
            System.out.println(exr.nextFloat());
    }
    
    public static void main(String[] args)
    {
        test1();
    }
}
