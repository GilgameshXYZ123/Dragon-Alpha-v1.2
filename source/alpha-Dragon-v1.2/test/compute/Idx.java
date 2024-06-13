/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package compute;

/**
 *
 * @author Gilgamesh
 */
public class Idx {
    
    static void compute1() {
        int x[] = {
            0, 1, 2, 8, 9, 10, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
            0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
            0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)
            0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
            0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24 ,25, 26, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
            0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 18, 20, 24, 25, 26, 27, 28,  0,  0,  0,  0,  0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)
            0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 5, tFW = 3, fhw_idx = (2, 0)
            0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,  0,  0,  0,  0,  0,//tFH = 5, tFW = 4, fhw_idx = (2, 1)
            0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36,//tFH = 5, tFW = 5, fhw_idx = (2, 2)
        };
        for(int i=0; i<x.length; i++) x[i] = x[i] >> 3;
        for(int i=0; i<9; i++) {
            for(int j=0; j<25; j++) {
                if(j !=0) System.out.print(", ");
                System.out.print(x[i*25 + j]);
            }
            System.out.println();
        }
        
    }
    
    public static void main(String[] args)
    {
        compute1();
    }    
}
