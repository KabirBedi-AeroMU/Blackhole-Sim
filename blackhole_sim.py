import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from vispy import app, scene
import sympy as sp
from PyQt5.QtWidgets import QInputDialog, QWidget

# =============================================================================
# 1. SYMPY INITIALIZATION 
# =============================================================================
print("[1/3] Initializing SymPy symbolic coordinate space...")
t, r, theta, phi = sp.symbols('t r theta phi')
M_sym = sp.symbols('M')
g_00 = -(1 - 2*M_sym/r)
g_11 = 1/(1 - 2*M_sym/r)
g_22 = r**2
g_33 = r**2 * sp.sin(theta)**2
print("      Symbolic spacetime metric verification complete.")

# =============================================================================
# 2. CUDA KERNEL (WITH ENHANCED EINSTEIN RING LENSING & QUASAR JETS)
# =============================================================================
print("[2/3] Compiling C++ PyCUDA parallel acceleration kernel...")
cuda_code = """
#include <math.h>

__global__ void render_blackhole(unsigned char *rgba, int width, int height, float M, float cam_height, float cam_dist, float ton_618, float m87, float zoom_factor, float optical_filter) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx_x >= width || idx_y >= height) return;
    
    float nx = (2.0f * idx_x / width) - 1.0f;
    float ny = (2.0f * idx_y / height) - 1.0f;
    float aspect = (float)width / (float)height;
    nx *= aspect;
    
    // RAY INITIATION
    float3 pos = make_float3(0.0f, cam_height, -cam_dist); 
    float3 vel = make_float3(nx, ny - (cam_height/cam_dist), 1.0f); 
    
    float v_mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
    vel.x /= v_mag; vel.y /= v_mag; vel.z /= v_mag;
    
    // Increased raymarching step size and max steps to allow massive zoom-outs
    float dt = 0.08f * M; 
    float3 color = make_float3(0.0f, 0.0f, 0.0f); 
    
    bool hit_blackhole = false;
    
    // 2000 steps ensures light can travel extreme distances when zoomed out!
    for (int step = 0; step < 2000; step++) {
        float r2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z;
        float r = sqrt(r2);
        
        // EVENT HORIZON
        if (r <= 2.0f * M) {
            hit_blackhole = true;
            break; 
        }
        
        // ESCAPE Optimization 
        if (r > fmaxf(25.0f * M, cam_dist + 50.0f)) {
            break; 
        }
        
        // --- REALISTIC 3D QUASAR PLASMA JETS ---
        // Jets show for TON 618 always, and for M87 when the optical filter is OFF
        if (ton_618 > 0.5f || (m87 > 0.5f && optical_filter < 0.5f)) {
            float abs_y = fabs(pos.y);
            if (abs_y > 2.0f * M) { 
                float rho = sqrt(pos.x*pos.x + pos.z*pos.z);
                
                float dist_factor = fminf(1.0f, (abs_y - 2.0f * M) / (25.0f * M));
                float base_radius = 0.15f * M + 0.12f * (abs_y - 2.0f * M);
                
                if (m87 > 0.5f) {
                    base_radius *= 1.3f; // M87 jets are massive and wider
                }
                
                float tx = pos.x / M; float ty = pos.y / M; float tz = pos.z / M;
                float noise = sin(tx*15.0f) * cos(tz*15.0f) * sin(ty*10.0f);
                noise += 0.5f * sin(tx*25.0f - ty*20.0f);
                
                float cone_radius = base_radius * (1.0f + 1.2f * dist_factor * noise);
                
                if (rho < cone_radius) {
                    float normalized_r = rho / cone_radius;
                    float core_density = exp(-4.0f * normalized_r * normalized_r);
                    float scatter_density = core_density * (1.0f - 0.5f * dist_factor * fabs(noise));
                    
                    float height_fade = exp(-(abs_y - 2.0f * M) / (18.0f * M)); 
                    float tip_cutoff = fmaxf(0.0f, 1.0f - (abs_y / (45.0f * M))); 
                    
                    float jet_intensity = scatter_density * height_fade * tip_cutoff * 1.5f * dt;
                    
                    if (jet_intensity > 0.0f) {
                        if (m87 > 0.5f) {
                            // M87: Blinding white/blue-white jets
                            color.x += jet_intensity * 0.9f; 
                            color.y += jet_intensity * 0.9f; 
                            color.z += jet_intensity * 1.0f; 
                        } else {
                            // TON 618: Deep cyan jets
                            color.x += jet_intensity * 0.35f; 
                            color.y += jet_intensity * 0.85f; 
                            color.z += jet_intensity * 1.5f;  
                        }
                    }
                }
            }
        }
        
        // ACCRETION DISK CROSSING
        float disk_inner = 2.6f * M; 
        float disk_outer = 14.0f * M;
        
        float next_y = pos.y + vel.y * dt;
        if ((pos.y >= 0.0f && next_y <= 0.0f) || (pos.y <= 0.0f && next_y >= 0.0f)) {
            
            if (fabs(vel.y) > 0.001f) {
                float t_cross = -pos.y / vel.y;
                float cross_x = pos.x + vel.x * t_cross;
                float cross_z = pos.z + vel.z * t_cross;
                float cross_r = sqrt(cross_x*cross_x + cross_z*cross_z);
                
                if (cross_r >= disk_inner && cross_r <= disk_outer) {
                    float u = (disk_outer - cross_r) / (disk_outer - disk_inner);
                    float intensity = u * u; 
                    float subtle_ring = 0.9f + 0.1f * sin(cross_r * 1.5f / M);
                    
                    if (ton_618 > 0.5f) {
                        color.x += intensity * 1.2f * subtle_ring;
                        color.y += (intensity * intensity) * 1.8f * subtle_ring; 
                        color.z += intensity * 2.5f * subtle_ring; 
                    } else if (m87 > 0.5f) {
                        // THE M87 PROFESSOR EXPLANATION: RELATIVISTIC DOPPLER BEAMING
                        float doppler = 1.0f - 0.85f * (cross_x / cross_r); 
                        
                        if (optical_filter < 0.5f) {
                            // FILTER OFF: Blindingly bright core, almost star-like
                            color.x += intensity * 4.0f * doppler * subtle_ring; 
                            color.y += intensity * 3.5f * doppler * subtle_ring; 
                            color.z += intensity * 3.0f * doppler * subtle_ring;  
                        } else {
                            // FILTER ON: The famous dim orange radio image
                            color.x += intensity * 1.8f * doppler * subtle_ring; 
                            color.y += (intensity * intensity) * 0.6f * doppler * subtle_ring; 
                            color.z += (intensity * intensity * intensity) * 0.1f * doppler * subtle_ring;  
                        }
                    } else {
                        color.x += intensity * 1.3f * subtle_ring;           
                        color.y += (intensity * intensity) * 0.8f * subtle_ring; 
                        color.z += (intensity * intensity * intensity) * 0.4f * subtle_ring;  
                    }
                }
            }
        }
        
        // RELATIVISTIC GRAVITY BENDING MATH
        float3 h = make_float3(
            pos.y * vel.z - pos.z * vel.y,
            pos.z * vel.x - pos.x * vel.z,
            pos.x * vel.y - pos.y * vel.x
        );
        float h2 = h.x*h.x + h.y*h.y + h.z*h.z;
        float r5 = r2 * r2 * r;
        
        float acc_factor = -1.5f * M * h2 / r5;
        
        vel.x += acc_factor * pos.x * dt;
        vel.y += acc_factor * pos.y * dt;
        vel.z += acc_factor * pos.z * dt;
        
        float v_norm = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        vel.x /= v_norm; vel.y /= v_norm; vel.z /= v_norm;
        
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
    }
    
    // --- NEW BACKGROUND STARFIELD LOGIC ---
    if (!hit_blackhole) {
        float band = exp(-fabs(vel.y) * 12.0f) * 0.15f;
        
        float size_adj = 1.0f / zoom_factor;
        
        float seed = vel.x * 213.2f + vel.y * 812.9f + vel.z * 453.1f;
        float noise = fabs(sin(seed) * 43758.5453f);
        noise = noise - floor(noise);
        
        float noise_thresh = 1.0f - (0.0005f * size_adj);
        if (noise > noise_thresh) {
            float star_b = (noise - noise_thresh) * (4000.0f / size_adj); 
            color.x += star_b * 0.85f;
            color.y += star_b * 0.9f;
            color.z += star_b * 1.0f;
        }
        
        const int NUM_STARS = 6;
        float3 stars[NUM_STARS];
        stars[0] = make_float3(0.0f, 0.0f, 1.0f);      
        stars[1] = make_float3(0.5f, 0.4f, 0.76f);     
        stars[2] = make_float3(-0.6f, 0.3f, 0.74f);    
        stars[3] = make_float3(0.4f, -0.5f, 0.76f);    
        stars[4] = make_float3(-0.45f, -0.55f, 0.7f);  
        stars[5] = make_float3(0.7f, -0.1f, 0.7f);     

        float3 star_colors[NUM_STARS];
        star_colors[0] = make_float3(1.0f, 0.9f, 0.75f); 
        star_colors[1] = make_float3(0.7f, 0.85f, 1.0f); 
        star_colors[2] = make_float3(1.0f, 1.0f, 1.0f);  
        star_colors[3] = make_float3(1.0f, 0.85f, 0.6f); 
        star_colors[4] = make_float3(0.8f, 0.9f, 1.0f);  
        star_colors[5] = make_float3(1.0f, 0.95f, 0.9f); 

        float star_sizes[NUM_STARS];
        star_sizes[0] = 0.9996f;  
        star_sizes[1] = 0.99993f; 
        star_sizes[2] = 0.9997f;  
        star_sizes[3] = 0.99995f; 
        star_sizes[4] = 0.99985f; 
        star_sizes[5] = 0.9999f;  

        for(int i=0; i<NUM_STARS; i++) {
            float3 s_dir = stars[i];
            float l = sqrt(s_dir.x*s_dir.x + s_dir.y*s_dir.y + s_dir.z*s_dir.z);
            s_dir.x /= l; s_dir.y /= l; s_dir.z /= l;
            
            float dot_p = vel.x*s_dir.x + vel.y*s_dir.y + vel.z*s_dir.z;
            
            float base_size = 1.0f - star_sizes[i];
            float adjusted_size = base_size * (1.0f / zoom_factor);
            float thresh = 1.0f - adjusted_size;
            
            if (dot_p > thresh) { 
                float brightness_mult = 1.0f / adjusted_size;
                float intensity = (dot_p - thresh) * brightness_mult * 1.5f;
                
                color.x += intensity * star_colors[i].x;
                color.y += intensity * star_colors[i].y;
                color.z += intensity * star_colors[i].z;
            }
        }

        color.x += band;
        color.y += band * 0.8f;
        color.z += band * 0.9f;
    }
    
    // Output RGB clamping
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);
    
    int pixel_idx = (idx_y * width + idx_x) * 4;
    rgba[pixel_idx]     = (unsigned char)(color.x * 255.0f);
    rgba[pixel_idx + 1] = (unsigned char)(color.y * 255.0f);
    rgba[pixel_idx + 2] = (unsigned char)(color.z * 255.0f);
    rgba[pixel_idx + 3] = 255; 
}
"""

mod = SourceModule(cuda_code)
render_kernel = mod.get_function("render_blackhole")
print("      GPU compiled source module successfully loaded.")

# =============================================================================
# 3. VISPY WINDOW INTERACTIVE ENVIRONMENT & HUD
# =============================================================================
print("[3/3] Setting up VisPy interactive workspace canvas window...")

class BlackHoleCanvas(scene.SceneCanvas):
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys=None, size=(1024, 768), show=True, title="Advanced Black Hole Lensing Simulator")
        self.unfreeze()
        self.width, self.height = self.size
        
        self.img_array = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.view = self.central_widget.add_view()
        self.image = scene.visuals.Image(self.img_array, parent=self.view.scene)
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, self.width), y=(0, self.height))
        
        # Physics Parameters 
        self.mass = 20.0
        self.cam_height = 16.0 
        self.cam_dist = 280.0 
        self.ton_618_mode = 0.0 
        self.m87_mode = 0.0
        self.zoom_factor = 1.0 
        self.is_dead = False # Track if the satellite crashed
        self.optical_filter = 1.0 # Tracks the J key filter
        
        # HUD OVERLAY (Now anchored to the CENTER horizontally)
        self.mass_text = scene.visuals.Text(
            text=f"Solar Mass: {self.mass:,.1f} M",
            color='#00ffff', 
            font_size=18,
            bold=True,
            anchor_x='center', # Centers the text origin horizontally
            anchor_y='center',
            parent=self.scene,
            pos=(self.width // 2, 40) # Positioned dead center at the top
        )
        
        # DEATH SCREEN TEXT 
        self.death_text = scene.visuals.Text(
            text="Satellite got caught in magnetic field of Blackhole\nPress ENTER to restart",
            color='#ff3333', 
            font_size=28,
            bold=True,
            anchor_x='center', 
            anchor_y='center', 
            parent=self.scene,
            pos=(self.size[0] // 2, self.size[1] // 2)
        )
        self.death_text.visible = False
        
        # OPTICAL FILTER TEXT (Now anchored to the CENTER horizontally, just below Mass Text)
        self.filter_text = scene.visuals.Text(
            text="Optical filter is turned on",
            color='#aaaaaa', 
            font_size=18,
            bold=True,
            anchor_x='center', 
            anchor_y='center', 
            parent=self.scene,
            pos=(self.width // 2, 80) # Positioned right underneath the Solar Mass text
        )
        self.filter_text.visible = False
        
        self.update_simulation()
        self.freeze()
        
    def trigger_death(self):
        self.is_dead = True
        self.death_text.pos = (self.size[0] // 2, self.size[1] // 2)
        self.img_array.fill(0)
        self.img_array[:, :, 3] = 255 
        self.image.set_data(self.img_array)
        self.mass_text.visible = False
        self.filter_text.visible = False
        self.death_text.visible = True
        self.update()
        
    def update_simulation(self):
        block_size = (16, 16, 1)
        grid_size = (int(np.ceil(self.width / 16)), int(np.ceil(self.height / 16)), 1)
        
        safe_M = 20.0
        scale_factor = safe_M / self.mass
        
        render_cam_height = self.cam_height * scale_factor
        render_cam_dist = self.cam_dist * scale_factor

        render_kernel(
            cuda.InOut(self.img_array),
            np.int32(self.width),
            np.int32(self.height),
            np.float32(safe_M),
            np.float32(render_cam_height),
            np.float32(render_cam_dist),
            np.float32(self.ton_618_mode),
            np.float32(self.m87_mode),
            np.float32(self.zoom_factor),
            np.float32(self.optical_filter),
            block=block_size, grid=grid_size
        )
        self.image.set_data(self.img_array)
        
        # Keep updating position to the center of the screen when resized!
        self.mass_text.pos = (self.size[0] // 2, 40)
        self.filter_text.pos = (self.size[0] // 2, 80)
        
        if self.ton_618_mode == 1.0:
            self.mass_text.text = f"Solar Mass: {self.mass:,.1f} M\nWARNING: TON 618 QUASAR DETECTED"
            self.mass_text.color = '#ff33ff' 
            self.filter_text.visible = False
        elif self.m87_mode == 1.0:
            self.mass_text.text = f"Solar Mass: {self.mass:,.1f} M\nTARGET: M87 (EHT OBSERVATION)"
            self.mass_text.color = '#ffaa00'
            self.filter_text.visible = True
            if self.optical_filter == 1.0:
                self.filter_text.text = "Optical filter is turned on"
                self.filter_text.color = '#aaaaaa'
            else:
                self.filter_text.text = "Optical filter is turned off"
                self.filter_text.color = '#ff3333'
        else:
            self.mass_text.text = f"Solar Mass: {self.mass:,.1f} M"
            self.mass_text.color = '#00ffff'
            self.filter_text.visible = False
            
        self.update()
        
    def on_key_press(self, event):
        if self.is_dead:
            if event.key.name in ('Enter', 'Return'):
                self.mass = 20.0
                self.cam_height = 16.0
                self.cam_dist = 280.0
                self.ton_618_mode = 0.0
                self.m87_mode = 0.0
                self.zoom_factor = 1.0
                self.is_dead = False
                self.death_text.visible = False
                self.mass_text.visible = True
                self.update_simulation()
            return

        min_dist = self.mass * 3.0 
        max_height = self.cam_dist * 0.7 

        if event.text.lower() == 'i':
            dummy_widget = QWidget()
            text, ok = QInputDialog.getText(dummy_widget, 'Solar Mass Override', 'Enter precise Solar Mass (or target name):')
            if ok and text:
                val = text.strip().lower()
                if val in ['ton 618', 'ton618']:
                    self.mass = 66000000000.0 # 66 Billion
                    self.ton_618_mode = 1.0
                    self.m87_mode = 0.0
                    self.cam_dist = self.mass * 8.0 
                    self.cam_height = self.mass * 0.4
                    self.zoom_factor = 1.0 
                    self.optical_filter = 1.0
                elif val in ['m87', 'm 87', 'm-87']:
                    self.mass = 6500000000.0 # 6.5 Billion (M87)
                    self.ton_618_mode = 0.0
                    self.m87_mode = 1.0
                    self.cam_dist = self.mass * 12.0 
                    self.cam_height = self.mass * 1.5
                    self.zoom_factor = 1.0 
                    self.optical_filter = 1.0
                else:
                    try:
                        new_mass = float(val)
                        if new_mass > 0:
                            self.mass = new_mass
                            self.ton_618_mode = 0.0 
                            self.m87_mode = 0.0
                            self.cam_dist = max(new_mass * 8.0, 16.0) 
                            self.cam_height = new_mass * 0.4
                            self.zoom_factor = 1.0 
                            self.optical_filter = 1.0
                    except ValueError:
                        print("Invalid Input! Please enter a valid number.")
            self.update_simulation()
            
        elif event.text.lower() == 'w':
            step = max(2.0, self.mass * 0.05) 
            self.mass += step 
            self.update_simulation()
        elif event.text.lower() == 'j':
            # Toggle the optical filter state
            self.optical_filter = 0.0 if self.optical_filter == 1.0 else 1.0
            self.update_simulation()
        elif event.text.lower() == 's':
            step = max(2.0, self.mass * 0.05)
            self.mass = max(2.0, self.mass - step) 
            self.update_simulation()
        elif event.text.lower() == 'z': # Zoom In
            step = max(10.0, self.cam_dist * 0.05)
            self.cam_dist = max(min_dist, self.cam_dist - step) 
            self.cam_height = max(-self.cam_dist * 0.7, min(self.cam_dist * 0.7, self.cam_height))
            self.zoom_factor = max(0.4, self.zoom_factor * 0.95) 
            self.update_simulation()
        elif event.text.lower() == 'x': # Zoom Out
            step = max(10.0, self.cam_dist * 0.05)
            self.cam_dist += step 
            self.zoom_factor = min(2.5, self.zoom_factor * 1.05) 
            self.update_simulation()
        elif event.key.name == 'Up':
            step = max(4.0, self.cam_dist * 0.02)
            self.cam_height += step 
            if abs(self.cam_height) > max_height:
                self.trigger_death()
            else:
                self.update_simulation()
        elif event.key.name == 'Down':
            step = max(4.0, self.cam_dist * 0.02)
            self.cam_height -= step 
            if abs(self.cam_height) > max_height:
                self.trigger_death()
            else:
                self.update_simulation()
        elif event.text.lower() == 'r':
            self.mass = 20.0
            self.cam_height = 16.0
            self.cam_dist = 280.0
            self.ton_618_mode = 0.0
            self.m87_mode = 0.0
            self.zoom_factor = 1.0
            self.optical_filter = 1.0
            self.update_simulation()

if __name__ == '__main__':
    print("\n>>> System Status: Ready. Running execution pipeline loop...")
    print(">>> CONTROLS:")
    print("    [ I ] Open Input Dialog Box (Type 'TON 618' or 'M87')")
    print("    [ J ] Toggle Optical Filter (M87 Mode)")
    print("    [Z]/[X] Zoom IN / Zoom OUT")
    print("    [W]/[S] Mass UP / Mass DOWN")
    print("    [Up/Down Arrow] Adjust Camera Tilt")
    app.use_app('PyQt5')
    canvas = BlackHoleCanvas()
    app.run()