/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package swizzle;

import java.util.HashSet;

/**
 *
 * @author Gilgamesh
 */
public class Java1 
{
    public static void test1() {
        int gridX = 6, gridY = 12;
        int[][] BX = new int[gridY][gridX];
        int[][] BY = new int[gridY][gridX];
      
        HashSet<Integer> set = new HashSet<>();
        for(int by0=0; by0<gridY; by0++) {
            for(int bx0=0; bx0<gridX; bx0++) {
                int bidx = by0 * gridX + bx0;
                int sbidx = bidx >> 2, rbidx = bidx & 3;
                int sgridX = (gridX + 1) >> 1;
                int sby = sbidx / sgridX, sbx = sbidx % sgridX;
                int rby = rbidx >> 1, rbx = rbidx & 1;
                
                int by = (sby << 1) + rby;
                int bx = (sbx << 1) + rbx;
               
                BY[by][bx] = by0;
                BX[by][bx] = bx0;
                
                System.out.format("(%d, %d, %d, %d)", bx, by, bx0, by0);
                set.add(by * gridX + bx);
            }
            System.out.println();
        }
        System.out.println("count = " + set.size());
        
        for(int by=0; by<gridY; by++) {
            for(int bx=0; bx<gridX; bx++) {
                int bx0 = BX[by][bx];
                int by0 = BY[by][bx];
                int bidx = by0*gridX + bx0;
                
                System.out.format("%d, ", bidx);
            }
            System.out.println();
        }
    }
    
    public static void main(String[] args)
    {
        test1();
    }
    
}
