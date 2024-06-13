/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Shuffle 
{
    public static void test1() {
        ThreadLocalRandom.current();
        
        SimpleTimer timer = SimpleTimer.clock();
        int[] arr = Vector.random_int_vector(1280000);
        Arrays.sort(arr);
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
    }
    
    public static void main(String[] args) {
        test1();
    }
}
