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
public class Index 
{
    public static void test1() {
        for(int ty=0; ty<16; ty++) {
            for(int tx=0; tx<16; tx++) {
                int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//8 * 32
                int GIdx = ((uy & 1) + ((uy >> 3) << 1));//8
                int DIdx = ((uy & 7) >> 1);//4
                
                int threadIdx = tx + ty * 16;
                int laneIdx = threadIdx % 32;
                int warpIdx = threadIdx / 32;
                System.out.println("," + ux + ", " + warpIdx + ">\t <" + laneIdx + "," + uy + ">\t:" + GIdx + ", " + DIdx);
                if(uy != laneIdx) throw new RuntimeException();
                if(ux != warpIdx) throw new RuntimeException();
            }
            System.out.println();
        }   
         
    }
    
    public static void main(String[] args) {
        test1();
    }
}
