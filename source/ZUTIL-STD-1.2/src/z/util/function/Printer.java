/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.function;

import java.io.PrintStream;

/**
 *
 * @author dell
 */
public interface Printer<T> 
{
    public void println(PrintStream out, T val);
}
