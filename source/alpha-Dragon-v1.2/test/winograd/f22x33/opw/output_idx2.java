/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f22x33.opw;

/**
 *
 * @author Gilgamesh
 */
public class output_idx2 
{
    public static void idx1() {
        for(int ty=0; ty<8; ty++){
            for(int tx=0; tx<32 ;tx++) {
                int Idx = (ty << 5) + tx;//ty * 32 + tx
                int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
                int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;
                System.out.format("(%2d, %2d: %2d, %2d)", ty, tx, ux, GIdx);
            }
            System.out.println();
        }
    }
    
    public static void idx2() 
    {
//        for(int ty=0; ty<8; ty++) {
//            System.out.print("\nty = " + ty);
//            for(int tx=0; tx<32; tx++){
//               int ux = (ty << 1) + (tx > 15 ? 1 : 0);
//               int uy = (tx & 15);
//               int XIdx = ((uy & 1) + ((uy >> 3) << 1) + (tx > 15 ? 1 : 0) * 16);
//               int GIdx = ((uy & 7) >> 1);
//               System.out.format("(%2d, %2d, %2d)",  tx, XIdx, GIdx);
//            }
//        }
         
        for(int tx=0; tx<32; tx++){
            int uy = (tx & 15);
            int XIdx = ((uy & 1) + ((uy >> 3) << 1));
            int GIdx = ((uy & 7) >> 1);
            System.out.format("\n(%d, %d, %d),", tx, XIdx, GIdx); 
        
        }
    }
    
    
    
    public static void main(String[] args)
    {
        idx2();
    }
}
