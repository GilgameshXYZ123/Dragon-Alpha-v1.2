/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f22x33;

/**
 *
 * @author Gilgamesh
 */
public class WMat 
{
    final static float A[][] = new float[][] {
        {1,  0},
        {1,  1},
        {1, -1},
        {0, -1}
    };
    
    final static float AT[][] = new float[][] {
        {1, 1,  1,  0},
        {0, 1, -1, -1}
    };
    
    final static float G[][] = new float[][] {
        {   1,     0,    0},
        {0.5f,  0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {   0,    0,     1}, 
    };
    
    final static float GT[][] = new float[][] {
        {1, 0.5f,  0.5f, 0},
        {0, 0.5f, -0.5f, 0},
        {0, 0.5f,  0.5f, 1}
    };
    
    final static float B[][] = new float[][] {
        { 1, 0,  0,  0},
        { 0, 1, -1,  1},
        {-1, 1,  1,  0},
        {0,  0,  0, -1}
    };
    
    final static float BT[][] = new float[][] {
        {1,  0, -1,  0},
        {0,  1,  1,  0},
        {0, -1,  1,  0},
        {0,  1,  0, -1},
    };
}
