/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.ui;

import java.awt.AWTException;
import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Window;
import java.awt.FlowLayout;
import java.awt.Font;
import static java.awt.Font.BOLD;
import static java.awt.Font.ITALIC;
import static java.awt.Font.PLAIN;
import java.awt.Graphics;
import java.awt.GraphicsEnvironment;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Point;
import java.awt.SystemTray;
import java.awt.TrayIcon;
import java.awt.event.ActionEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.JSlider;
import javax.swing.JFileChooser;
import javax.swing.JColorChooser;
import javax.swing.JDesktopPane;
import javax.swing.JLayeredPane;
import javax.swing.JSplitPane;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JFrame;
import javax.swing.border.Border;


/**
 *
 * @author Gilgamesh
 */
public final class JUI 
{
    public static final JUI jui = new JUI();
    public final FontHandler font =  FontHandler.font;
    private JUI() {}

    public JFrame show_frame(Component con) {
        JFrame frame = this.frame();
        frame.add(con);
        frame.setSize(con.getWidth(), con.getHeight());
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        this.screen_center(frame);
        frame.setVisible(true);
        return frame;
    }
    
    //<editor-fold defaultstate="collapsed" desc="class: RoundLineBorder">
    public static class RoundLineBorder implements Border {
	private final Color color;
        private final int arc_size;

	public RoundLineBorder(Color color, int arc_size) { 
            this.color = color; 
            this.arc_size = arc_size;
        }

        @Override public Insets getBorderInsets(Component c) { return new Insets(0, 0, 0, 0); }
        @Override public boolean isBorderOpaque() { return false; }

	@Override
	public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
            g.setColor(color);
            g.drawRoundRect(0, 0, c.getWidth() - 1, c.getHeight() - 1, arc_size, arc_size);
	}   
    }
    //</editor-fold>
    public RoundLineBorder round_line_boarder(Color color, int arc_size) { return new RoundLineBorder(color, arc_size); }
    
    public void screen_center(Window window) { window.setLocationRelativeTo(null); }
    
    public static class MouseDrag_click implements MouseListener {
            Point startPoint;
            Component comp;
            
            MouseDrag_click(Component comp) { this.comp = comp; }
            
            @Override
            public void mousePressed(MouseEvent e) {
                int button = e.getButton();
                if(button == MouseEvent.BUTTON1) {
                    Point clickPoint = e.getLocationOnScreen();
                    Point compPoint = comp.getLocation();
                    int x = clickPoint.x - compPoint.x;
                    int y = clickPoint.y - compPoint.y;
                    startPoint = new Point(x, y);
                }
                else startPoint = null;
            }

            @Override public void mouseClicked(MouseEvent e) { }
            @Override public void mouseReleased(MouseEvent e) { }
            @Override public void mouseEntered(MouseEvent e) { }
            @Override public void mouseExited(MouseEvent e) { }
    }
    public void mouse_drag(Component comp) {
        MouseDrag_click click = new MouseDrag_click(comp);
        comp.addMouseListener(click);
        comp.addMouseMotionListener(new MouseMotionListener() {
            @Override
            public void mouseDragged(MouseEvent e) {
                Point startPoint = click.startPoint;
                if( startPoint != null) {
                    Point currentPoint = e.getLocationOnScreen();
                    int x = currentPoint.x - startPoint.x;
                    int y = currentPoint.y - startPoint.y;
                    comp.setLocation(x , y);
                }
            }

            @Override public void mouseMoved(MouseEvent e) { }
        });
    }
    public void mouse_drag(Component src, Component dst) {
        MouseDrag_click click = new MouseDrag_click(dst);
        src.addMouseListener(click);
        src.addMouseMotionListener(new MouseMotionListener() {
            @Override
            public void mouseDragged(MouseEvent e) {
                Point startPoint = click.startPoint;
                if( startPoint != null) {
                    Point currentPoint = e.getLocationOnScreen();
                    int x = currentPoint.x - startPoint.x;
                    int y = currentPoint.y - startPoint.y;
                    dst.setLocation(x , y);
                }
            }

            @Override public void mouseMoved(MouseEvent e) { }
        });
    }
    
    public TrayIcon system_tray(JButton button, Window window, BufferedImage icon)  {
        TrayIcon tryIcon = new TrayIcon(icon);
        button.addActionListener((ActionEvent e) -> {
            try { 
                SystemTray.getSystemTray().add(tryIcon); 
                window.setVisible(false);  
            }  catch(AWTException ex) { throw new RuntimeException(ex); }
        });
        tryIcon.addActionListener((ActionEvent e) -> {
            SystemTray.getSystemTray().remove(tryIcon); 
            window.setVisible(true);
        });
        return tryIcon;
    }
    
    static final char[] HEX = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'  };
    public String toString(Color color) {
        StringBuilder sb = new StringBuilder(64).append("0x");
        int r = color.getRed();   sb.append(HEX[r >> 4]).append(HEX[r & 15]);
        int g = color.getGreen(); sb.append(HEX[g >> 4]).append(HEX[g & 15]);
        int b = color.getBlue();  sb.append(HEX[b >> 4]).append(HEX[b & 15]);
        return sb.toString();
    }
    
    public void set_width(Component c, int width) { c.setSize(width, c.getHeight()); }
    public void set_height(Component c, int height) { c.setSize(c.getWidth(), height); }
    
    public Component left (Component ref, Component c, int gap) { c.setLocation(ref.getX() - c.getWidth() - 4, ref.getY()); return c; }
    public Component right(Component ref, Component c, int gap) { c.setLocation(ref.getX() + ref.getWidth() + gap, ref.getY()); return c; }
    public Component below(Component ref, Component c, int gap) { c.setLocation(ref.getX(), ref.getY() + ref.getHeight() + gap); return c; }
    
    //<editor-fold defaultstate="collapsed" desc="layout">
    public FlowLayout layout_flow(int align, int ygap, int xgap) { return new FlowLayout(align, xgap, ygap); }
    public FlowLayout layout_flow(int align) { return new FlowLayout(align); }
    public FlowLayout layout_flow() { return new FlowLayout(); }
    
    public BorderLayout layout_border(int ygap, int xgap) { return new BorderLayout(ygap, xgap); }
    public BorderLayout layout_border() { return new BorderLayout(); }

    public GridLayout layout_grid(int rows, int cols, int ygap, int xgap) { return new GridLayout(rows, cols, xgap, ygap); }
    public GridLayout layout_grid(int rows, int cols) { return new GridLayout(rows, cols); }
    
    public CardLayout layout_card(int ygap, int xgap) { return new CardLayout(xgap, ygap); }
    public CardLayout layout_card() { return new CardLayout(); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="basic element">
    public JButton button(Icon icon, String text) { return new JButton(text); }
    public JButton button(String text) { return new JButton(text); }
    public JButton button(Icon icon) { return new JButton(icon); }
    public JButton button() { return new JButton(); }
    
    public JLabel label(String text) { return new JLabel(text); }
    public JLabel label(Icon icon) { return new JLabel(icon); }
    public JLabel label() { return new JLabel(); }
    
    public JTextField text_field(String text, int column) { return new JTextField(text, column); }
    public JTextField text_field(String text) { return new JTextField(text); }
    public JTextField text_field(int column) { return new JTextField(column); }
    public JTextField text_field() { return new JTextField(); }
    
    public JTextArea text_area(String text, int rows, int cols) { return new JTextArea(text, rows, cols); }
    public JTextArea text_area(String text) { return new JTextArea(text); }
    public JTextArea text_area(int rows, int cols) { return new JTextArea(rows, cols); }
    public JTextArea text_area() { return new JTextArea(); }
    
    public JTextPane text_pane() { return new JTextPane(); }

    public JSlider slider(int min, int max, int value) { return new JSlider(min, max, value); }
    public JSlider slider(int min, int max) { return new JSlider(min, max); }
    public JSlider slider(int orientation) { return new JSlider(orientation); }
    public JSlider slider() { return new JSlider(); }
    
    public JFileChooser file_chooser(File dir) { return new JFileChooser(dir); }
    public JFileChooser file_chooser(String dir) { return new JFileChooser(dir); }
    public JFileChooser file_chooser() { return new JFileChooser(); }
    
    public JColorChooser color_chooser(Color init_color) { return new JColorChooser(init_color); }
    public JColorChooser color_chooser() { return new JColorChooser(); }
    //</editor-fold>
    
    public JDesktopPane pane_desktop() { return new JDesktopPane(); }
    public JLayeredPane pane_layered() { return new JLayeredPane(); }
    public JSplitPane pane_split() { return new JSplitPane(); }
    public JScrollPane pane_scroll() { return new JScrollPane(); }
    public JTabbedPane pane_tabbed() { return new JTabbedPane(); }
    public JFrame frame() { return new JFrame(); }
    
    public String[] font_names() { return GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames(); }
    
    public static final class FontHandler {
        public static final FontHandler font = new FontHandler();
        
        private FontHandler() {}      
        
        public Font italic(Font font) { return new Font(font.getFamily(), (font.isBold() ? BOLD | ITALIC : ITALIC), font.getSize()); }
        public Font bold(Font font) { return new Font(font.getFamily(), (font.isItalic() ? BOLD | ITALIC : BOLD), font.getSize()); }
        public Font bold_italic(Font font) { return new Font(font.getFamily(), BOLD | ITALIC, font.getSize()); }
        
        public Font Times_New_Roman(int size) { return new Font("Times New Roman", PLAIN, size); }
        public Font Arial(int size) { return new Font("Arial", PLAIN, size);  }
        public Font SongTi(int size) { return new Font("宋体", PLAIN, size); }
        public Font KaiTi(int size) { return new Font("楷体", PLAIN, size); }
    }
    
    public static final class ColorHandler {
        public static final ColorHandler color = new ColorHandler();
        
        private ColorHandler() {}
        
        public Color dark_gray() { return new Color(169, 169, 169); }
        public Color silver_gray() { return new Color(192, 192, 192); }
        public Color platinum_gray() { return new Color(229, 228, 226); }
        
        public Color white() { return Color.WHITE; }
        public Color vanilla_white() { return new Color(243, 229, 171); }
        public Color shell_white() { return new Color(255, 245, 238); }
        public Color ivory_white() { return new Color(255, 255, 240); }
       public Color light_ivory_white() { return new Color(255, 255, 249); }
        public Color pearl_white() { return new Color(226, 230, 210); }
        public Color rice_white() { return new Color(245, 245, 220); }
         
        public Color black() { return Color.BLACK; }
        public Color matte_black() { return new Color(40, 40, 43); }
        
        public Color blue() { return Color.BLUE; }
        public Color dark_blue() { return new Color(0, 0, 129); }
        public Color night_blue() { return new Color(25, 25, 112); }
        public Color iron_blue() { return new Color(70, 130, 180); }
        public Color egyptian_blue() {  return new Color(20, 52, 164); }
        public Color perwhinkle_blue() { return new Color(204, 204, 255); }
        public Color sky_blue() { return new Color(240, 255, 255); }
        public Color light_sky_blue() { return new Color(249, 255, 255); }
        
        public Color red() { return Color.RED; }
        public Color blood_red() { return new Color(74, 4, 4); }
    }
}
