/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import java.util.Arrays;
import z.util.math.Num;

/**
 *
 * @author Gilgamesh
 */
public class Prime 
{
    public static void prime_table() {
        int[] T =  Num.primeTable(512);
        for(int i=0; i<T.length; i++) {
            if(T[i] == 0) System.out.println(i);
        }
        
        System.out.println(Arrays.toString(T));
    }
    
    public static void main(String[] args) {
        prime_table();
    }
    
}
