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
import java.awt.Rectangle;
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
import javax.swing.plaf.basic.BasicScrollBarUI;
import z.util.lang.Lang;


/**
 *
 * @author Gilgamesh
 */
public final class JUI 
{
    public static final JUI jui = new JUI();
    public final FontHandler font =  FontHandler.font;
    public final ColorHandler clr = ColorHandler.clr;
    
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
    public Component bellow(Component ref, Component c, int gap) { c.setLocation(ref.getX(), ref.getY() + ref.getHeight() + gap); return c; }
    
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
    
    public static class BlueScrollBarUI extends BasicScrollBarUI {
        @Override
        protected void configureScrollBarColors() {
            
            ColorHandler color = ColorHandler.clr;
            
//            this.thumbColor = clr.eletric_blue();
//            this.thumbDarkShadowColor = clr.egyptian_blue();
//            this.thumbHighlightColor = clr.sky_blue();
            
            this.trackColor = color.sky_blue();
            this.trackHighlightColor = color.night_blue();
            this.thumbRect = new Rectangle(0, 20, 0, 20);
            super.configureScrollBarColors();
            
        }
    }
    
    public BlueScrollBarUI scrollBarUI_blue() { return new BlueScrollBarUI(); }
    
    //<editor-fold defaultstate="collapsed" desc="class: ColorHandler">
    public static final class ColorHandler {
        public static final ColorHandler clr = new ColorHandler();
        private ColorHandler() {}
        
        public Color alpha(Color c, float alpha) { return new Color(c.getRed(), c.getGreen(), c.getBlue(), (int)(alpha * 255)); }//[0-1]
        public Color alpha(Color c, int   alpha) { return new Color(c.getRed(), c.getGreen(), c.getBlue(), alpha); }//[0-255]
        
        //<editor-fold defaultstate="collapsed" desc="clr: gray">
        private static final Color gray = Color.GRAY;
        private static final Color dark_gray = new Color(169, 169, 169);
        private static final Color sliver_gray = new Color(192, 192, 192);
        private static final Color platinum_gray = new Color(229, 228, 226);
        private static final Color bronze_gray = new Color(129, 133, 137);
        
        public Color gray() { return gray; }
        public Color dark_gray() { return dark_gray; }
        public Color silver_gray() { return sliver_gray; }
        public Color platinum_gray() { return platinum_gray; }
        public Color bronze_gray() { return bronze_gray; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: white">
        private static final Color white = Color.WHITE;
        private static final Color vanilla_white = new Color(243, 229, 171);
        private static final Color shell_white = new Color(255, 245, 238);
        private static final Color ivory_white = new Color(255, 255, 240);
        private static final Color light_ivory_white = new Color(255, 255, 249);
        private static final Color pearl_white = new Color(226, 230, 210);
        private static final Color rice_white = new Color(245, 245, 220);
        
        public Color white() { return white; }
        public Color vanilla_white() { return vanilla_white; }
        public Color shell_white() { return shell_white; }
        public Color ivory_white() { return ivory_white; }
        public Color light_ivory_white() { return light_ivory_white; }
        public Color pearl_white() { return pearl_white; }
        public Color rice_white() { return rice_white; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: black">
        private static final Color black = Color.BLACK;
        private static final Color matte_black = new Color(40, 40, 43);
        private static final Color ink_black = new Color(52, 52, 52); 
        private static final Color licorice_black = new Color(27, 18, 18);
        private static final Color agate_black = new Color(53, 57, 53);
        
        public Color black() { return black; }
        public Color matte_black() { return matte_black; }
        public Color ink_black() { return ink_black; }
        public Color licorice_black() { return licorice_black; }
        public Color agate_black() { return agate_black; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: blue">
        private static final Color blue = Color.BLUE;
        private static final Color dark_blue = new Color(0, 0, 129);
        private static final Color night_blue = new Color(25, 25, 112);
        private static final Color iron_blue = new Color(70, 130, 180);
        private static final Color egyptian_blue = new Color(20, 52, 164);
        private static final Color perwhinkle_blue =  new Color(204, 204, 255);
        private static final Color sky_blue = new Color(240, 255, 255);
        private static final Color light_sky_blue = new Color(249, 255, 255);
        private static final Color cornflower_blue = new Color(100, 149, 237);
        private static final Color eletric_blue = new Color(125, 239, 255);;
        private static final Color royal_blue = new Color(64, 105, 225);
        private static final Color light_blue = new Color(172, 216, 230);
        private static final Color cyan = new Color(0, 255, 255);
        
        public Color blue() { return blue; }
        public Color dark_blue() { return dark_blue; }
        public Color night_blue() { return night_blue; }
        public Color iron_blue() { return iron_blue; }
        public Color egyptian_blue() {  return egyptian_blue; }
        public Color perwhinkle_blue() { return perwhinkle_blue; }
        public Color sky_blue() { return sky_blue; }
        public Color light_sky_blue() { return light_sky_blue; }
        public Color cornflower_blue() { return cornflower_blue; }
        public Color eletric_blue() { return eletric_blue; }
        public Color royal_blue() { return royal_blue; }
        public Color light_blue() { return light_blue; }
        public Color cyan() { return cyan; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: red">
        private static final Color red = Color.RED;
        private static final Color blood_red = new Color(74, 4, 4);
        private static final Color black_red = new Color(136, 8, 8);
        private static final Color scarlet_red = new Color(238, 75, 43);
        private static final Color bishop_red = new Color(196, 30, 58);
        private static final Color wine_red = new Color(129, 19, 49);
        private static final Color dark_red = new Color(139, 0, 0);
        private static final Color berry_red = new Color(227, 11, 92);
        private static final Color salmon_red = new Color(250, 128, 114);
        private static final Color masala_red = new Color(152, 104, 104);
                
        public Color red() { return red; }
        public Color blood_red() { return blood_red; }
        public Color black_red() { return black_red; }
        public Color scarlet_red() { return scarlet_red; }
        public Color bishop_red() { return bishop_red; }
        public Color wine_red() { return wine_red; }
        public Color dark_red() { return dark_red; }
        public Color berry_red() { return berry_red; }
        public Color salmon_red() { return salmon_red; }
        public Color masala_red() { return masala_red; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: yellow">
        private static final Color yellow = Color.YELLOW;
        private static final Color light_yellow = new Color(255, 234, 0);
        private static final Color canary_yellow = new Color(255, 255, 143);
        private static final Color cystal_yellow = new Color(228, 208, 10);
        private static final Color dark_yellow = new Color(139, 128, 0);
        private static final Color golden_yellow = new Color(255, 192, 0);
        private static final Color lemon_yellow = new Color(250, 250, 51);
        private static final Color jasmine_yellow = new Color(248, 222, 126);
        private static final Color mustard_yellow = new Color(255, 219, 88);
        
        public Color yellow() { return yellow; }
        public Color light_yellow() { return light_yellow; }
        public Color canary_yellow() { return canary_yellow; }
        public Color cystal_yellow() { return cystal_yellow; }
        public Color dark_yellow() { return dark_yellow; }
        public Color golden_yellow() { return golden_yellow; }
        public Color lemon_yellow() { return lemon_yellow; }
        public Color jasmine_yellow() { return jasmine_yellow; }
        public Color mastard_yellow() { return mustard_yellow; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="clr: green">
        private static final Color green = Color.GREEN;
        private static final Color light_green = new Color(170, 255, 0);
        private static final Color cadmium_green = new Color(9, 121, 105);
        private static final Color dark_green = new Color(2, 48, 32);
        private static final Color plant_green = new Color(79, 121, 66);
        private static final Color grass_green = new Color(124, 252, 0);
        private static final Color mountain_green = new Color(11, 218, 81);
        private static final Color hunter_green = new Color(53, 94, 59);
        private static final Color mint_green = new Color(152, 251, 152);
        private static final Color sea_green = new Color(46, 139, 87);
        
        public Color green() { return green; }
        public Color light_green() { return light_green; }
        public Color cadmium_green() { return cadmium_green; }
        public Color dark_green() { return dark_green; }
        public Color plant_green() { return plant_green; }
        public Color grass_green() { return grass_green; }
        public Color mountain_green() { return mountain_green; }
        public Color hunter_green() { return hunter_green; }
        public Color mint_green() { return mint_green; }
        public Color sea_green() { return sea_green; }
        //</editor-fold>
        
        public static final Color[] colors = {
            gray, dark_gray, sliver_gray, platinum_gray, bronze_gray,
            white, vanilla_white, shell_white, ivory_white, light_ivory_white, pearl_white, rice_white,
            black, matte_black, ink_black, licorice_black, agate_black,
            blue, dark_blue, night_blue, iron_blue, egyptian_blue, perwhinkle_blue, sky_blue, light_sky_blue, cornflower_blue, eletric_blue, royal_blue, light_blue, cyan,
            red, blood_red, black_red, scarlet_red, bishop_red, wine_red, dark_red, berry_red, salmon_red, masala_red,
            yellow, light_yellow, canary_yellow, cystal_yellow, dark_yellow, golden_yellow, lemon_yellow, 
            green, light_green, cadmium_green, dark_green, plant_green, grass_green, mountain_green, hunter_green, mint_green, sea_green
        };
        
        public Color random_color() { return Lang.exr().select(colors); }
    }
    //</editor-fold>
}
