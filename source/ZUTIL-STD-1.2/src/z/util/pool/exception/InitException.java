/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.exception;

/**
 *
 * @author dell
 */
public class InitException extends RuntimeException
{
    public static final String MESSAGE="PoolInitException: ";
    
    public InitException()
    {
        super();
    }
    public InitException(String msg)
    {
        super(MESSAGE+msg);
    }
}
