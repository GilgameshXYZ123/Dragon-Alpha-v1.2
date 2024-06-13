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
public class output_idx 
{
    public static void idx() {
        for(int ty=0; ty<8;  ty++) {
            for(int tx=0; tx<32; tx++) {
                int oc = ((tx / 16) * 16 + (ty % 4) * 4 + ty / 4);
                int n  = ((tx % 16) * 2);//2 element along n
                System.out.print("[" + oc + ", " + n + "], ");
            }
            System.out.println();
        }
    }
    
    public static void idx2(){
        for(int tx=0; tx<32; tx++) {
            int accum1 = 
                    ((tx % 8) / 2) * (32 + 2) + 
                    ((tx / 8) % 2) * 8 +
                    (tx % 2) + 
                    (tx / 16) * 2;
           System.out.println(accum1);
        }
    }
    
    public static void idx3() {
        for(int ty=0; ty<16; ty++) {
            for(int tx=0; tx<16; tx++) {
                int goc = ty; if(tx >= 8) goc += 16;
                int xj =  tx; if(ty >= 8) xj += 16;
                System.out.format("[(%2d, %2d), %2d, %2d], ", ty, tx, goc, xj);
            }
            System.out.println();
        }
    }
    
    public static void idx4() {
        for(int tx=0 ;tx<32; tx++) {
            int oc = (tx % 8) * 4;
            int gt = (tx / 4) * 4;
            System.out.format("(%d, %d)\n", oc, gt);
        }
    }
    
    
//[( 0,  0),  0,  0], [( 0,  1),  0,  1], [( 0,  2),  0,  2], [( 0,  3),  0,  3], [( 0,  4),  0,  4], [( 0,  5),  0,  5], [( 0,  6),  0,  6], [( 0,  7),  0,  7], [( 0,  8), 16,  8], [( 0,  9), 16,  9], [( 0, 10), 16, 10], [( 0, 11), 16, 11], [( 0, 12), 16, 12], [( 0, 13), 16, 13], [( 0, 14), 16, 14], [( 0, 15), 16, 15], 
//[( 1,  0),  1,  0], [( 1,  1),  1,  1], [( 1,  2),  1,  2], [( 1,  3),  1,  3], [( 1,  4),  1,  4], [( 1,  5),  1,  5], [( 1,  6),  1,  6], [( 1,  7),  1,  7], [( 1,  8), 17,  8], [( 1,  9), 17,  9], [( 1, 10), 17, 10], [( 1, 11), 17, 11], [( 1, 12), 17, 12], [( 1, 13), 17, 13], [( 1, 14), 17, 14], [( 1, 15), 17, 15], 
//[( 2,  0),  2,  0], [( 2,  1),  2,  1], [( 2,  2),  2,  2], [( 2,  3),  2,  3], [( 2,  4),  2,  4], [( 2,  5),  2,  5], [( 2,  6),  2,  6], [( 2,  7),  2,  7], [( 2,  8), 18,  8], [( 2,  9), 18,  9], [( 2, 10), 18, 10], [( 2, 11), 18, 11], [( 2, 12), 18, 12], [( 2, 13), 18, 13], [( 2, 14), 18, 14], [( 2, 15), 18, 15], 
//[( 3,  0),  3,  0], [( 3,  1),  3,  1], [( 3,  2),  3,  2], [( 3,  3),  3,  3], [( 3,  4),  3,  4], [( 3,  5),  3,  5], [( 3,  6),  3,  6], [( 3,  7),  3,  7], [( 3,  8), 19,  8], [( 3,  9), 19,  9], [( 3, 10), 19, 10], [( 3, 11), 19, 11], [( 3, 12), 19, 12], [( 3, 13), 19, 13], [( 3, 14), 19, 14], [( 3, 15), 19, 15], 
//[( 4,  0),  4,  0], [( 4,  1),  4,  1], [( 4,  2),  4,  2], [( 4,  3),  4,  3], [( 4,  4),  4,  4], [( 4,  5),  4,  5], [( 4,  6),  4,  6], [( 4,  7),  4,  7], [( 4,  8), 20,  8], [( 4,  9), 20,  9], [( 4, 10), 20, 10], [( 4, 11), 20, 11], [( 4, 12), 20, 12], [( 4, 13), 20, 13], [( 4, 14), 20, 14], [( 4, 15), 20, 15], 
//[( 5,  0),  5,  0], [( 5,  1),  5,  1], [( 5,  2),  5,  2], [( 5,  3),  5,  3], [( 5,  4),  5,  4], [( 5,  5),  5,  5], [( 5,  6),  5,  6], [( 5,  7),  5,  7], [( 5,  8), 21,  8], [( 5,  9), 21,  9], [( 5, 10), 21, 10], [( 5, 11), 21, 11], [( 5, 12), 21, 12], [( 5, 13), 21, 13], [( 5, 14), 21, 14], [( 5, 15), 21, 15], 
//[( 6,  0),  6,  0], [( 6,  1),  6,  1], [( 6,  2),  6,  2], [( 6,  3),  6,  3], [( 6,  4),  6,  4], [( 6,  5),  6,  5], [( 6,  6),  6,  6], [( 6,  7),  6,  7], [( 6,  8), 22,  8], [( 6,  9), 22,  9], [( 6, 10), 22, 10], [( 6, 11), 22, 11], [( 6, 12), 22, 12], [( 6, 13), 22, 13], [( 6, 14), 22, 14], [( 6, 15), 22, 15], 
//[( 7,  0),  7,  0], [( 7,  1),  7,  1], [( 7,  2),  7,  2], [( 7,  3),  7,  3], [( 7,  4),  7,  4], [( 7,  5),  7,  5], [( 7,  6),  7,  6], [( 7,  7),  7,  7], [( 7,  8), 23,  8], [( 7,  9), 23,  9], [( 7, 10), 23, 10], [( 7, 11), 23, 11], [( 7, 12), 23, 12], [( 7, 13), 23, 13], [( 7, 14), 23, 14], [( 7, 15), 23, 15], 
//[( 8,  0),  8, 16], [( 8,  1),  8, 17], [( 8,  2),  8, 18], [( 8,  3),  8, 19], [( 8,  4),  8, 20], [( 8,  5),  8, 21], [( 8,  6),  8, 22], [( 8,  7),  8, 23], [( 8,  8), 24, 24], [( 8,  9), 24, 25], [( 8, 10), 24, 26], [( 8, 11), 24, 27], [( 8, 12), 24, 28], [( 8, 13), 24, 29], [( 8, 14), 24, 30], [( 8, 15), 24, 31], 
//[( 9,  0),  9, 16], [( 9,  1),  9, 17], [( 9,  2),  9, 18], [( 9,  3),  9, 19], [( 9,  4),  9, 20], [( 9,  5),  9, 21], [( 9,  6),  9, 22], [( 9,  7),  9, 23], [( 9,  8), 25, 24], [( 9,  9), 25, 25], [( 9, 10), 25, 26], [( 9, 11), 25, 27], [( 9, 12), 25, 28], [( 9, 13), 25, 29], [( 9, 14), 25, 30], [( 9, 15), 25, 31], 
//[(10,  0), 10, 16], [(10,  1), 10, 17], [(10,  2), 10, 18], [(10,  3), 10, 19], [(10,  4), 10, 20], [(10,  5), 10, 21], [(10,  6), 10, 22], [(10,  7), 10, 23], [(10,  8), 26, 24], [(10,  9), 26, 25], [(10, 10), 26, 26], [(10, 11), 26, 27], [(10, 12), 26, 28], [(10, 13), 26, 29], [(10, 14), 26, 30], [(10, 15), 26, 31], 
//[(11,  0), 11, 16], [(11,  1), 11, 17], [(11,  2), 11, 18], [(11,  3), 11, 19], [(11,  4), 11, 20], [(11,  5), 11, 21], [(11,  6), 11, 22], [(11,  7), 11, 23], [(11,  8), 27, 24], [(11,  9), 27, 25], [(11, 10), 27, 26], [(11, 11), 27, 27], [(11, 12), 27, 28], [(11, 13), 27, 29], [(11, 14), 27, 30], [(11, 15), 27, 31], 
//[(12,  0), 12, 16], [(12,  1), 12, 17], [(12,  2), 12, 18], [(12,  3), 12, 19], [(12,  4), 12, 20], [(12,  5), 12, 21], [(12,  6), 12, 22], [(12,  7), 12, 23], [(12,  8), 28, 24], [(12,  9), 28, 25], [(12, 10), 28, 26], [(12, 11), 28, 27], [(12, 12), 28, 28], [(12, 13), 28, 29], [(12, 14), 28, 30], [(12, 15), 28, 31], 
//[(13,  0), 13, 16], [(13,  1), 13, 17], [(13,  2), 13, 18], [(13,  3), 13, 19], [(13,  4), 13, 20], [(13,  5), 13, 21], [(13,  6), 13, 22], [(13,  7), 13, 23], [(13,  8), 29, 24], [(13,  9), 29, 25], [(13, 10), 29, 26], [(13, 11), 29, 27], [(13, 12), 29, 28], [(13, 13), 29, 29], [(13, 14), 29, 30], [(13, 15), 29, 31], 
//[(14,  0), 14, 16], [(14,  1), 14, 17], [(14,  2), 14, 18], [(14,  3), 14, 19], [(14,  4), 14, 20], [(14,  5), 14, 21], [(14,  6), 14, 22], [(14,  7), 14, 23], [(14,  8), 30, 24], [(14,  9), 30, 25], [(14, 10), 30, 26], [(14, 11), 30, 27], [(14, 12), 30, 28], [(14, 13), 30, 29], [(14, 14), 30, 30], [(14, 15), 30, 31], 
//[(15,  0), 15, 16], [(15,  1), 15, 17], [(15,  2), 15, 18], [(15,  3), 15, 19], [(15,  4), 15, 20], [(15,  5), 15, 21], [(15,  6), 15, 22], [(15,  7), 15, 23], [(15,  8), 31, 24], [(15,  9), 31, 25], [(15, 10), 31, 26], [(15, 11), 31, 27], [(15, 12), 31, 28], [(15, 13), 31, 29], [(15, 14), 31, 30], [(15, 15), 31, 31], 

    public static void main(String[] args)
    {
        idx4();
    }
}
