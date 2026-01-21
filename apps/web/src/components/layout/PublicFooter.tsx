import { Link } from "react-router-dom";
import {
  WaveformIcon,
  WaveformDecoration,
} from "@/components/brand/WaveformIcon";

export function PublicFooter() {
  return (
    <footer className="border-t border-border bg-muted/30 py-12 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute bottom-0 right-0 opacity-[0.03] pointer-events-none">
        <WaveformDecoration className="h-48 w-auto text-primary" />
      </div>

      <div className="container relative">
        <div className="grid gap-8 md:grid-cols-4">
          <div>
            <Link to="/" className="flex items-center gap-2.5 mb-4 group">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary transition-transform group-hover:scale-105">
                <WaveformIcon className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="font-semibold text-foreground">
                Energy Monitor
              </span>
            </Link>
            <p className="text-sm text-muted-foreground leading-relaxed">
              AI-powered NILM technology for real-time appliance-level energy
              insights from a single meter.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-foreground mb-4">Product</h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <Link
                  to="/about"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  About NILM
                </Link>
              </li>
              <li>
                <Link
                  to="/docs"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Documentation
                </Link>
              </li>
              <li>
                <Link
                  to="/contact"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Contact Us
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-foreground mb-4">Support</h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <Link
                  to="/docs"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Help Center
                </Link>
              </li>
              <li>
                <Link
                  to="/contact"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Get in Touch
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>
            © {new Date().getFullYear()} Energy Monitor. All rights reserved.
          </p>
          <p className="flex items-center gap-1.5">
            <span>Powered by</span>
            <span className="font-medium text-primary">NILM AI</span>
          </p>
        </div>
      </div>
    </footer>
  );
}
