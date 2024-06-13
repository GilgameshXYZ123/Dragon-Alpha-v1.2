/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f4x3;

/**
 *
 * @author Gilgamesh
 */
public class A2 
{
    //<1> M = G*p <*> B^T*s
    //m0 = (24*s0 - 30*s2 + 6*s4)         * p0
    //m1 = ( 16*s1 + 16*s2 - 4*s3 - 4*s4) * (p0 + p1 + p2)
    //m2 = (-16*s1 + 16*s2 - 4*s3 - 4*s4) * (p0 - p1 + p2)
    //m3 = (-2*s1 - s2 + 2*s3 + s4)       * (p0 + 2*p1 + 4*p2)
    //m4 =( 2*s1 - s2 - 2*s3 + s4)       * (p0 - 2*p1 + 4*p2)
    //m5 = (96*s1 - 120*s3 + 24*s5)       *  p2
    
   
    static float[][] AT = { //<2> h = A^T * M
        {1, 1,  1, 1,  1, 0},//h0 = m0 + m1 + m2 +   m3 +   m4
        {0, 1, -1, 2, -2, 0},//h1 =      m1 - m2 + 2*m3 - 2*m4
        {0, 1,  1, 4,  4, 0},//h2 =      m1 + m2 + 4*m3 + 4*m4
        {0, 1, -1, 8, -8, 1} //h3 =      m1 - m2 + 8*m3 - 8*m4 + m5 
    };
    
}
