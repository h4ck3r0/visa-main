// Global app state
let app = {
    sidebar: {
        collapsed: false,
        active: false
    },
    fab: {
        active: false
    },
    currentPage: 'home',
    notifications: [],
    modals: []
};

// Initialize the application
function initializeApp() {
    console.log('🚀 Initializing Visa Assistant Pro...');
    
    // Hide loading screen after a delay
    setTimeout(() => {
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
        }
    }, 1500);

    // Initialize components
    initializeSidebar();
    initializeFAB();
    initializeNavigation();
    initializeToasts();
    initializeModals();
    initializeAnimations();
    
    // Set current page active
    setActivePage();
    
    console.log('✅ App initialized successfully');
}

// Sidebar functionality
function initializeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const menuBtn = document.getElementById('menuBtn');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', toggleSidebar);
    }
    
    if (menuBtn) {
        menuBtn.addEventListener('click', toggleSidebar);
    }
    
    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768) {
            if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
                if (app.sidebar.active) {
                    toggleSidebar();
                }
            }
        }
    });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    
    if (window.innerWidth <= 768) {
        // Mobile: toggle visibility
        app.sidebar.active = !app.sidebar.active;
        sidebar.classList.toggle('active', app.sidebar.active);
    } else {
        // Desktop: toggle collapsed state
        app.sidebar.collapsed = !app.sidebar.collapsed;
        sidebar.classList.toggle('collapsed', app.sidebar.collapsed);
    }
    
    // Animate sidebar toggle
    sidebar.style.transform = app.sidebar.collapsed || (window.innerWidth <= 768 && !app.sidebar.active) 
        ? 'translateX(-100%)' 
        : 'translateX(0)';
    
    // Add animation class
    sidebar.classList.add('animating');
    setTimeout(() => sidebar.classList.remove('animating'), 300);
}

// Floating Action Button
function initializeFAB() {
    const fabMain = document.getElementById('fabMain');
    const fabMenu = document.getElementById('fabMenu');
    
    if (fabMain && fabMenu) {
        fabMain.addEventListener('click', toggleFAB);
        
        // Handle FAB item clicks
        const fabItems = fabMenu.querySelectorAll('.fab-item');
        fabItems.forEach(item => {
            item.addEventListener('click', handleFABAction);
        });
        
        // Close FAB when clicking outside
        document.addEventListener('click', (e) => {
            if (!fabMain.contains(e.target) && !fabMenu.contains(e.target)) {
                if (app.fab.active) {
                    toggleFAB();
                }
            }
        });
    }
}

function toggleFAB() {
    const fabMenu = document.getElementById('fabMenu');
    const fabMain = document.getElementById('fabMain');
    
    app.fab.active = !app.fab.active;
    fabMenu.classList.toggle('active', app.fab.active);
    
    // Rotate FAB button
    fabMain.style.transform = app.fab.active ? 'scale(1.1) rotate(45deg)' : 'scale(1) rotate(0deg)';
    
    // Animate menu items
    const items = fabMenu.querySelectorAll('.fab-item');
    items.forEach((item, index) => {
        setTimeout(() => {
            item.style.transform = app.fab.active 
                ? 'translateY(0) scale(1)' 
                : 'translateY(20px) scale(0.9)';
            item.style.opacity = app.fab.active ? '1' : '0';
        }, index * 50);
    });
}

function handleFABAction(e) {
    const action = e.currentTarget.getAttribute('data-action');
    
    switch (action) {
        case 'new-application':
            navigateTo('/application');
            break;
        case 'upload-document':
            navigateTo('/documents');
            break;
        case 'ask-ai':
            navigateTo('/chat');
            break;
        default:
            console.log('Unknown FAB action:', action);
    }
    
    toggleFAB(); // Close FAB menu
}

// Navigation
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('data-page');
            const href = link.getAttribute('href');
            
            if (href && href !== '#') {
                navigateTo(href);
            }
        });
    });
}

function navigateTo(url) {
    // Add loading animation
    showPageTransition();
    
    // Navigate after animation
    setTimeout(() => {
        window.location.href = url;
    }, 300);
}

function showPageTransition() {
    const overlay = document.createElement('div');
    overlay.className = 'page-transition-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        z-index: 9998;
        opacity: 0;
        transition: opacity 0.3s ease-out;
    `;
    
    document.body.appendChild(overlay);
    
    // Trigger animation
    setTimeout(() => {
        overlay.style.opacity = '0.9';
    }, 10);
    
    // Remove after navigation
    setTimeout(() => {
        if (overlay.parentNode) {
            overlay.parentNode.removeChild(overlay);
        }
    }, 600);
}

function setActivePage() {
    const path = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === path) {
            link.classList.add('active');
        }
    });
}

// Toast notifications
function initializeToasts() {
    // Auto-remove existing toasts after 5 seconds
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => {
        setTimeout(() => {
            removeToast(toast);
        }, 5000);
    });
}

function showToast(message, type = 'info', duration = 5000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">
                <i class="fas fa-${getToastIcon(type)}"></i>
            </div>
            <div class="toast-message">${message}</div>
            <button class="toast-close" onclick="removeToast(this.parentNode.parentNode)">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    }, 10);
    
    // Auto remove
    setTimeout(() => {
        removeToast(toast);
    }, duration);
    
    return toast;
}

function removeToast(toast) {
    if (!toast || !toast.parentNode) return;
    
    toast.style.transform = 'translateX(100%)';
    toast.style.opacity = '0';
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 300);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Modals
function initializeModals() {
    const modalOverlay = document.getElementById('modalOverlay');
    
    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                closeModal();
            }
        });
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modalOverlay.classList.contains('active')) {
                closeModal();
            }
        });
    }
}

function showModal(content, options = {}) {
    const overlay = document.getElementById('modalOverlay');
    const modalContent = document.getElementById('modalContent');
    
    if (!overlay || !modalContent) return;
    
    modalContent.innerHTML = content;
    overlay.classList.add('active');
    
    // Focus trap
    const focusableElements = modalContent.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    if (focusableElements.length > 0) {
        focusableElements[0].focus();
    }
}

function closeModal() {
    const overlay = document.getElementById('modalOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// Animations and effects
function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-bounce-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe cards and other elements
    const animatedElements = document.querySelectorAll('.card, .btn, .form-group');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
    
    // Parallax effect for floating shapes
    window.addEventListener('scroll', () => {
        const shapes = document.querySelectorAll('.shape');
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        shapes.forEach((shape, index) => {
            const speed = 0.2 + (index * 0.1);
            shape.style.transform = `translateY(${rate * speed}px)`;
        });
    });
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-box input');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Ctrl/Cmd + B for sidebar toggle
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
    }
});

// Handle online/offline status
window.addEventListener('online', () => {
    showToast('Connection restored', 'success');
});

window.addEventListener('offline', () => {
    showToast('Connection lost. Working offline.', 'warning');
});

// Export functions for global use
window.app = app;
window.showToast = showToast;
window.showModal = showModal;
window.closeModal = closeModal;
window.navigateTo = navigateTo;