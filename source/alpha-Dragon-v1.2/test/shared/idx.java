/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shared;

import java.util.Arrays;
import z.util.math.Sort;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class idx 
{
    static Integer[] A = {
        0,   1,  2,  3,  4,  5,  6,  7,  
        8,   9, 10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
    };
    
    static Integer[] A2 = {
        0, 7,  8, 15, 16, 20, 24, 28,
        1, 6,  9, 14, 17, 21, 25, 29,
        2, 5, 10, 13, 18, 22, 26, 30,
        3, 4, 11, 12, 19, 23, 27, 31
    };
    
    static Integer[] A3 = {
         0,  1,  2,  3, 4,  5,   6,  7,
        15, 14, 13, 12, 11, 10,  9,  8,
        16, 17, 18, 19, 20, 21, 22, 23, 
        31, 30, 29, 28, 27, 26, 25, 24,
    };
    
    static Integer[] A4 = {
        0, 4,  8, 12, 16, 20, 24, 28,
        1, 5,  9, 13, 17, 21, 25, 29,
        2, 6, 10, 14, 18, 22, 26, 30,
        3, 7, 11, 15, 19, 23, 27, 31,
    };
    
    static Integer[] A5 = {
         0,  2,  4,  6,  8, 10, 12, 14,
         1,  3,  5,  7,  9, 11, 13, 15,
        16, 18, 20, 22, 24, 26, 28, 30,
        17, 19, 21, 23, 25, 27, 29, 31
    };
    
    static Integer A6[] = {//STD
        0, 1,  8,  9, 16, 17, 24, 25,
        2, 3, 10, 11, 18, 19, 26, 27,
        4, 5, 12, 13, 20, 21, 28, 29,
        6, 7, 14, 15, 22, 23, 30, 31
    };
    
    static Integer A7[] = { 
         0,  1,  2,  3, 16, 17, 18, 19,
         4,  5,  6,  7, 20, 21, 22, 23,
         8,  9, 10, 11, 24, 25, 26, 27,
        12, 13, 14, 15, 28, 29, 30, 31,   
    };
     
    static Integer A8[] = {//STD
         0,  1,  2,  3,  4,  5,  6,  7,
         9,  8, 11, 10, 13, 12, 15, 14,
        18, 19, 16, 17, 22, 23, 20, 21,
        27, 26, 25, 24, 31, 30, 29, 28
    };
    
    static Integer A9[] = {//STD
        3, 2, 11, 10, 19, 18, 27, 26,
        7, 6, 15, 15, 23, 22, 31, 30,
        1, 0,  9,  8, 17, 16, 25, 25,
        5, 4, 13, 12, 21, 20, 29, 28
    };
    
    static Integer[] A10 = {
        0,   1,  2,  3,  4,  5,  6,  7,//tx: 0 ->  7, ty = 0, i0
        8,   9, 10, 11, 12, 13, 14, 15,//tx: 8 -> 16, ty = 0, i1
        16, 17, 18, 19, 20, 21, 22, 23,//tx: 0 ->  7, ty = 1, i2
        24, 25, 26, 27, 28, 29, 30, 31,//tx: 8 -> 16, ty = 1, i3
    };
    
    static Integer[] A11 = {
        0,   3,  1,  5,  2,  6,  4,  7,
        9,  8, 11, 10, 13, 12, 15, 14,
        18, 19, 16, 17, 22, 23, 20, 21,
        27, 26, 25, 24, 31, 30, 29, 28
    };
    
    public static void to_shared_idx() {
        Sort.shuffle_sort(A);
        A = A8;
        
        int XIdx[] = new int[32];
        int GIdx[] = new int[32];
        
        for(int i=0; i<4; i++)
        for(int j=0; j<8; j++) {
            int elem = A[i*8 + j];
            XIdx[elem] = i;//[0 - 4]
            GIdx[elem] = j;//[0 - 8]
        }
        
        System.out.println("__constant__ int XI[] = { " + Vector.toString(XIdx) + " };");
        System.out.println("__constant__ int XK[] = { " + Vector.toString(GIdx) + " };");
    }
    
    public static void std_idx() {
//        int XIdx[] = new int[32];
//        int GIdx[] = new int[32];
//        
//        for(int uy=0; uy<32; uy++) {
//            GIdx[uy] = ((uy & 1) + ((uy >> 3) << 1));
//            XIdx[uy] = ((uy & 7) >> 1);
//        }

        int XIdx[] = { 0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3 };
        int GIdx[] = { 0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,4,5,4,5,4,5,4,5,6,7,6,7,6,7,6,7 };

        
        int[] K = new int[32];
        for(int i=0; i<4; i++)
        for(int j=0; j<8; j++)
        {
            int uy = i*8 + j;
            int gidx = GIdx[uy];
            int xidx = XIdx[uy];
            int idx = gidx + xidx * 8;
            K[idx] = uy;
        }
        
        System.out.println("__constant__ int XIDXS[] = { " + Vector.toString(XIdx) + " };");
        System.out.println("__constant__ int GIDXS[] = { " + Vector.toString(GIdx) + " };");
        System.out.println("__constant__ int GIDXS[] = { " + Vector.toString(K) + " };");
    }    
    
    
    public static void main(String[] args)
    {
//        std_idx();
//        to_shared_idx();
        
//       System.out.format("%7Ef\n", 1.0 * 40063 / 288);
        
//        System.out.println(Math.log(134217728*2) / Math.log(4));
//        
        for(int i=1; i<=9; i++)
        {
            double a = Math.pow(4, 15 - i);
            float v = (float) ((a * 1.0) / 160810650);
            System.out.format("%7Ef\n", v);
            
        }
        
    }
    
    
}
