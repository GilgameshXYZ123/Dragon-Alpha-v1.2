/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package random;

import java.util.Arrays;
import java.util.Random;
import z.util.math.ExRandom;
import z.util.math.Num;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class r2 
{
    Random r = new ExRandom();
    
    public static boolean isPrime(int x) {
        int len = (int) Math.sqrt(x);
        for(int i=2; i<=len; i++) 
            if(x % i == 0) return false;
        return true;
    }
    
    public static void test1() {
        int x = (1 << 21) - 1;
        System.out.print(x);
    }
    
    
    public static void main(String[] args)
    {
        int x = (1 << 30) - 1;
//        int x = 251403917;
        System.out.println(x);
        System.out.println(251403917);
        System.out.println(Arrays.toString(Num.primeFactors(x)));
        System.out.println(isPrime(x));
        
    }
}
