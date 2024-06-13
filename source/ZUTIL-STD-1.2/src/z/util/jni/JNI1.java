/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.jni;

import z.util.math.vector.Vector;

/**
 *
 * @author dell
 */
public class JNI1 
{
    static
    {
        System.load("D:\\virtual disc Z-Gilgamesh\\Gilgamesh java2\\ZUTIL-STD-1.1\\src\\z\\util\\jni\\jni1.dll");
    }
    public native static void add(double[] arr);
    
    public static void main(String[] args)
    {
        double[] arr=Vector.random_double_vector(10);
        Vector.println(arr);
        add(arr);
        Vector.println(arr);
    }
}
