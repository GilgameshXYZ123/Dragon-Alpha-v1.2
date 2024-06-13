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
public class shuffle 
{
    public static void idx(int FW)
    {
        for(int fw=1; fw <= FW; fw++) {
            int x = (3 * fw - 3) & 3;
            int y = (3 * fw - 2) & 3;
            int z = (3 * fw - 1) & 3;
            int w = (3 * fw    ) & 3;
            int update = ((fw - 1) & 3);
            
            System.out.println(x + ", " + y + ", " + z + ", " + w);
            System.out.println(update);
        }
    }
    
    public static void main(String[] args)
    {
        idx(3);
    }
}
