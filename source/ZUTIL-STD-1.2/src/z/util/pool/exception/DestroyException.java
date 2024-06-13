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
public class DestroyException extends RuntimeException
{
    public static final String MSG="PoolDestroyException: ";
    
    public DestroyException()
    {
        super();
    }
    public DestroyException(String msg)
    {
        super(MSG+msg);
    }
    public DestroyException(Throwable e)
    {
        super(e);
    }
}
