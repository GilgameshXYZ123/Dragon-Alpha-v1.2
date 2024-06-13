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
public class CreateException extends RuntimeException
{
    public static final String MSG="PoolCreateException: ";
    
    public CreateException()
    {
        super();
    }
    public CreateException(Throwable e)
    {
        super(e);
    }
    public CreateException(String msg)
    {
        super(MSG+msg);
    }
}
