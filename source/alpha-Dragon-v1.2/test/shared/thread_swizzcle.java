/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shared;

import java.util.HashSet;

/**
 *
 * @author Gilgamesh
 */
public class thread_swizzcle 
{
    public static class KV implements Comparable {
        KV(int ty, int tx) {
            this.ty = ty;
            this.tx = tx;
        } 
        
        int ty;
        int tx;
        
        @Override
        public int compareTo(Object o) {
            if(!(o instanceof KV)) return -1;
            
            KV kv = (KV) o;
            
            int div0 = ty - kv.ty;
            if(div0 != 0) return div0;
            
            int div1 = tx - kv.tx;
            return div1;
        }
    }
    
    public static void idx() {
        int blockDim_y = 3;
        int blockDim_x = 7;
        
        int ymax = 0;
        int xmax = 0;
        
        HashSet<KV> set = new HashSet<>();
        
        for(int by=0; by < blockDim_y; by++) {
            for(int bx=0; bx < blockDim_x; bx++)
            {
                int bidx = by * blockDim_x + bx;
                
                int by0 = (bidx / (blockDim_x * 2)) * 2 + (bidx & 1);
                int bx0 = (bidx % (blockDim_x * 2)) / 2;
                
                
                System.out.format("(%d, %d), ", by0, bx0);
                
                if(ymax < by0) ymax = by0;
                if(xmax < bx0) xmax = bx0;
                set.add(new KV(by0, bx0));
                
                //int bidx0 = by0 * blockDim_x + bx0;
                //System.out.print(bidx0 + ", ");
            }
            System.out.println();
        }
        
        System.out.println(set.size());
        System.out.println("ymax = " + ymax);
        System.out.println("xmax = " + xmax);
    }
    
    public static void main(String[] args)
    {
        idx();
    }
      
}
