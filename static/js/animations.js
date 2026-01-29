// Advanced animations and effects

class AnimationManager {
    constructor() {
        this.animations = new Map();
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupHoverEffects();
        this.setupLoadingAnimations();
        this.setupParticleSystem();
    }

    setupScrollAnimations() {
        // Enhanced intersection observer with multiple thresholds
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const element = entry.target;
                const animationType = element.dataset.animation || 'fadeInUp';
                
                if (entry.isIntersecting) {
                    this.playAnimation(element, animationType);
                }
            });
        }, {
            threshold: [0, 0.1, 0.5, 1],
            rootMargin: '0px 0px -50px 0px'
        });

        // Observe elements with data-animation attribute
        document.querySelectorAll('[data-animation]').forEach(el => {
            observer.observe(el);
        });
    }

    playAnimation(element, type) {
        element.style.opacity = '0';
        element.style.transform = this.getInitialTransform(type);
        
        // Force reflow
        element.offsetHeight;
        
        // Apply animation
        element.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        element.style.opacity = '1';
        element.style.transform = 'none';
        
        // Add completion class
        setTimeout(() => {
            element.classList.add('animation-complete');
        }, 800);
    }

    getInitialTransform(type) {
        const transforms = {
            fadeInUp: 'translateY(60px)',
            fadeInDown: 'translateY(-60px)',
            fadeInLeft: 'translateX(-60px)',
            fadeInRight: 'translateX(60px)',
            zoomIn: 'scale(0.8)',
            rotateIn: 'rotate(-180deg) scale(0.8)',
            slideInUp: 'translateY(100%)',
            bounceIn: 'scale(0.3)',
            flipInX: 'perspective(400px) rotateX(-90deg)',
            flipInY: 'perspective(400px) rotateY(-90deg)'
        };
        return transforms[type] || transforms.fadeInUp;
    }

    setupHoverEffects() {
        // Enhanced card hover effects
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                this.animateCardHover(e.target, true);
            });
            
            card.addEventListener('mouseleave', (e) => {
                this.animateCardHover(e.target, false);
            });
        });

        // Button ripple effect
        document.querySelectorAll('.btn').forEach(button => {
            button.addEventListener('click', (e) => {
                this.createRipple(e);
            });
        });
    }

    animateCardHover(card, isEntering) {
        const transform = isEntering 
            ? 'translateY(-10px) scale(1.02)' 
            : 'translateY(0) scale(1)';
        
        const shadow = isEntering
            ? '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
            : '0 10px 15px -3px rgba(0, 0, 0, 0.1)';

        card.style.transform = transform;
        card.style.boxShadow = shadow;
        card.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    }

    createRipple(event) {
        const button = event.currentTarget;
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;

        const ripple = document.createElement('div');
        ripple.classList.add('ripple');
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: scale(0);
            pointer-events: none;
            animation: ripple 0.6s ease-out;
        `;

        // Add ripple styles if not already added
        if (!document.querySelector('#ripple-styles')) {
            const style = document.createElement('style');
            style.id = 'ripple-styles';
            style.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(2);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        button.appendChild(ripple);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }

    setupLoadingAnimations() {
        // Skeleton loading animation
        this.createSkeletonStyles();
        
        // Loading spinner variations
        this.createSpinnerVariations();
    }

    createSkeletonStyles() {
        if (document.querySelector('#skeleton-styles')) return;

        const style = document.createElement('style');
        style.id = 'skeleton-styles';
        style.textContent = `
            .skeleton {
                background: linear-gradient(90deg, 
                    rgba(255,255,255,0.05) 25%, 
                    rgba(255,255,255,0.1) 50%, 
                    rgba(255,255,255,0.05) 75%
                );
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s ease-in-out infinite;
                border-radius: 4px;
            }
            
            @keyframes skeleton-loading {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            
            .skeleton-text {
                height: 1rem;
                margin: 0.5rem 0;
            }
            
            .skeleton-title {
                height: 1.5rem;
                width: 60%;
                margin: 1rem 0;
            }
            
            .skeleton-avatar {
                width: 3rem;
                height: 3rem;
                border-radius: 50%;
            }
        `;
        document.head.appendChild(style);
    }

    createSpinnerVariations() {
        if (document.querySelector('#spinner-styles')) return;

        const style = document.createElement('style');
        style.id = 'spinner-styles';
        style.textContent = `
            .spinner {
                display: inline-block;
                position: relative;
            }
            
            .spinner-dots {
                width: 40px;
                height: 40px;
            }
            
            .spinner-dots::before,
            .spinner-dots::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--primary-color);
                animation: spinner-dots 1.2s ease-in-out infinite;
            }
            
            .spinner-dots::before {
                animation-delay: -0.16s;
            }
            
            @keyframes spinner-dots {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            
            .spinner-pulse {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: var(--primary-color);
                animation: spinner-pulse 1.5s ease-in-out infinite;
            }
            
            @keyframes spinner-pulse {
                0%, 100% { opacity: 1; transform: scale(0); }
                50% { opacity: 0.5; transform: scale(1); }
            }
        `;
        document.head.appendChild(style);
    }

    setupParticleSystem() {
        this.particles = [];
        this.createParticleCanvas();
        this.startParticleAnimation();
    }

    createParticleCanvas() {
        const canvas = document.createElement('canvas');
        canvas.id = 'particle-canvas';
        canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.3;
        `;
        
        document.body.appendChild(canvas);
        
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resizeCanvas();
        
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    startParticleAnimation() {
        // Create initial particles
        for (let i = 0; i < 50; i++) {
            this.particles.push(this.createParticle());
        }
        
        this.animateParticles();
    }

    createParticle() {
        return {
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            size: Math.random() * 3 + 1,
            speedX: (Math.random() - 0.5) * 2,
            speedY: (Math.random() - 0.5) * 2,
            opacity: Math.random() * 0.5 + 0.2,
            hue: Math.random() * 60 + 200 // Blue to purple range
        };
    }

    animateParticles() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach((particle, index) => {
            // Update position
            particle.x += particle.speedX;
            particle.y += particle.speedY;
            
            // Wrap around screen
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.y > this.canvas.height) particle.y = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            
            // Draw particle
            this.ctx.save();
            this.ctx.globalAlpha = particle.opacity;
            this.ctx.fillStyle = `hsl(${particle.hue}, 70%, 60%)`;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();
            
            // Draw connections to nearby particles
            this.particles.forEach((otherParticle, otherIndex) => {
                if (index !== otherIndex) {
                    const distance = Math.sqrt(
                        Math.pow(particle.x - otherParticle.x, 2) +
                        Math.pow(particle.y - otherParticle.y, 2)
                    );
                    
                    if (distance < 100) {
                        const opacity = (100 - distance) / 100 * 0.2;
                        this.ctx.save();
                        this.ctx.globalAlpha = opacity;
                        this.ctx.strokeStyle = `hsl(${particle.hue}, 70%, 60%)`;
                        this.ctx.lineWidth = 1;
                        this.ctx.beginPath();
                        this.ctx.moveTo(particle.x, particle.y);
                        this.ctx.lineTo(otherParticle.x, otherParticle.y);
                        this.ctx.stroke();
                        this.ctx.restore();
                    }
                }
            });
        });
        
        requestAnimationFrame(() => this.animateParticles());
    }

    // Utility methods for triggering animations
    showElement(element, animation = 'fadeInUp') {
        element.style.display = 'block';
        element.style.opacity = '0';
        element.style.transform = this.getInitialTransform(animation);
        
        requestAnimationFrame(() => {
            this.playAnimation(element, animation);
        });
    }

    hideElement(element, callback) {
        element.style.transition = 'all 0.3s ease-out';
        element.style.opacity = '0';
        element.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            element.style.display = 'none';
            if (callback) callback();
        }, 300);
    }

    morphElement(element, newContent, animation = 'zoomIn') {
        this.hideElement(element, () => {
            element.innerHTML = newContent;
            this.showElement(element, animation);
        });
    }
}

// Page transition animations
class PageTransitions {
    constructor() {
        this.init();
    }

    init() {
        // Intercept navigation for smooth transitions
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href]');
            if (link && this.shouldIntercept(link)) {
                e.preventDefault();
                this.navigate(link.href);
            }
        });
    }

    shouldIntercept(link) {
        const href = link.getAttribute('href');
        return href && 
               !href.startsWith('#') && 
               !href.startsWith('http') && 
               !href.startsWith('mailto') && 
               !href.startsWith('tel') &&
               !link.hasAttribute('target');
    }

    navigate(url) {
        this.playExitAnimation(() => {
            window.location.href = url;
        });
    }

    playExitAnimation(callback) {
        const pageContent = document.querySelector('.page-content');
        const overlay = this.createTransitionOverlay();
        
        // Fade out page content
        pageContent.style.transition = 'all 0.3s ease-out';
        pageContent.style.opacity = '0';
        pageContent.style.transform = 'translateY(20px)';
        
        // Show overlay
        setTimeout(() => {
            overlay.style.opacity = '1';
        }, 100);
        
        // Navigate
        setTimeout(callback, 400);
    }

    createTransitionOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'page-transition';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            z-index: 9999;
            opacity: 0;
            transition: opacity 0.3s ease-out;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
        `;
        
        overlay.innerHTML = `
            <div class="transition-content">
                <div class="transition-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                </div>
                <div>Loading...</div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        return overlay;
    }
}

// Initialize advanced animations
const animationManager = new AnimationManager();
const pageTransitions = new PageTransitions();

// Export for global use
window.animationManager = animationManager;
window.pageTransitions = pageTransitions;