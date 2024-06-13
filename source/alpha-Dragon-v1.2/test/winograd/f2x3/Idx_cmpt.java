/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f2x3;

import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author Gilgamesh
 */
public class Idx_cmpt 
{
    public static void index() 
    {
        for(int uy=0; uy<64; uy++)
        {
            int DIdx = ((uy & 1) + ((uy >> 4) << 1));//0 -> 7, j 8*8 = 64
            int GIdx = ((uy & 15) >> 1)             ;//0 -> 7, oc
            
            if(uy % 8 == 0) System.out.println();
            System.out.format("(%d, %d)", GIdx, DIdx);
        }
    }
    
    public static void index2() 
    {
        for(int ty=0; ty<16; ty++) {
            
            for(int tx=0; tx<16; tx++)
            {
                int elements_per_thread = 16;
                int element_id = (ty << 6) + tx;
                int lane_id = element_id / elements_per_thread;
                int element_id_in_thread = element_id % elements_per_thread;
                int shared_c = lane_id % 8;
                int shared_s = lane_id / 8;
                int row = (shared_c & 1) | ((shared_c >> 1) & 2);
                int col = ((shared_c << 1) & 4) | shared_s ^ row;
                System.out.format("(%d, %d), ", row, col);
            }
            
            System.out.println();
        }
    }
    
    public static void index3() 
    {
        HashSet<Integer> set = new HashSet<>();
        for(int ty=0; ty<16; ty++) {
            for(int tx=0; tx<16; tx++)
            {
                int tid = ty * 16 + tx;
                int tid128 = tid + 128;
                int as = ((tid128 >> 4) | ((tid >> 1) & 7));
                int bs = (((tid & 112) >> 3) | (tid & 1));
//                int readAs = ((tid128 >> 4) | ((tid >> 1) & 7)) << 4;
//                int readBs = (((tid & 112) >> 3) | (tid & 1)) << 4;
                
                System.out.format("(%2d, %2d), ", as, bs);
                set.add(as*16 + bs);
            }
            
            System.out.println();
        }
       
        System.out.println(set.size());
    }
    
    
    public static void main(String[] args)
    {
        index3();
    }
}
