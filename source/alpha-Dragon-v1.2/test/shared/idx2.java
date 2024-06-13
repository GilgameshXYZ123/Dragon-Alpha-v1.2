/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shared;

/**
 *
 * @author Gilgamesh
 */
public class idx2 
{
    public static void test1()
    {
        int[][] Vk = new int[8][32];
        int[][] Vi = new int[8][32];
        
        for(int ty=0; ty<16; ty++) {
            for(int tx=0; tx<16; tx++) {
                int Ds_k0 = (tx & 7);
                int Ds_i0 = (ty << 1) + (tx / 8);
                
                int Ds_k1 = (ty & 7);
                int Ds_i1 = (tx << 1) + (ty / 8);
//                int Ds_k1 = (tx & 7);
//                int Ds_i1 = (ty << 1) + (tx / 8);
                
                Vk[Ds_k1][Ds_i1] = Ds_k0;
                Vi[Ds_k1][Ds_i1] = Ds_i0;       
            }
        }
        
        
        for(int k=0; k<8; k++)
        {
            for(int i=0; i<32; i++)
            {
                System.out.format("(i: %2d, k: %2d), ", Vi[k][i], Vk[k][i]);
            }
            System.out.println();
        }
        
    }
    
    public static void test2() 
    {
        for(int ty=0; ty<16; ty++) {
            for(int tx=0; tx<16; tx++)
            {
                int Ds_k = (tx & 7);
                int Ds_i = (ty << 1) + (tx / 8);
            
                System.out.format("(%2d, %2d), ", Ds_k, Ds_i);
            }
            
            System.out.println();
        }
        
        
    }
    

    public static void main(String[] args)
    {
        test2(); 
    }
}
