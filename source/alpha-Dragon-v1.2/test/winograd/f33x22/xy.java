
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f33x22;

/**
 *
 * @author Gilgamesh
 */
public class xy 
{
    public static void find_xy() {
        for(int x=1; x<=100; x++)
        for(int y=1; y<=100; y++) {
            boolean flag1 = (2*x + 1) == (3*y);
            int IH = 3*y - 1;
            int OH = 2*x;
            if(flag1) System.out.format("(IH, OH) = (%d, %d)\n", IH, OH);
        }
    }
    
    public static void main(String[] args)
    {
        find_xy();
    }
}
