import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { UserPlus, Shield, Users, Mail, Edit2, Loader2, Trash2, Clock } from 'lucide-react';
import { toast } from 'sonner';
import { NILMPanel } from '@/components/nilm/NILMPanel';
import { WaveformDecoration } from '@/components/brand/WaveformIcon';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { formatDistanceToNow } from 'date-fns';

type Role = 'admin' | 'member' | 'viewer';
type Status = 'active' | 'pending' | 'expired' | 'cancelled';

interface TeamMember {
  id: string;
  email: string;
  display_name: string | null;
  role: Role;
  status: Status;
  avatar_url: string | null;
}

interface Invitation {
  id: string;
  email: string;
  role: Role;
  status: Status;
  created_at: string;
  expires_at: string;
}

const roleConfig = {
  admin: {
    label: 'Admin',
    className: 'bg-destructive/10 text-destructive border-destructive/20',
    description: 'Full access to all features, user management, and settings.',
  },
  member: {
    label: 'Member',
    className: 'bg-primary/10 text-primary border-primary/20',
    description: 'Can view and analyze data, generate reports.',
  },
  viewer: {
    label: 'Viewer',
    className: 'bg-muted text-muted-foreground border-border',
    description: 'Read-only access to dashboards and reports.',
  },
};

const statusConfig = {
  active: {
    label: 'Active',
    className: 'bg-energy-success/10 text-energy-success border-energy-success/20',
  },
  pending: {
    label: 'Pending',
    className: 'bg-energy-warning-bg text-energy-warning border-energy-warning/20',
  },
  expired: {
    label: 'Expired',
    className: 'bg-muted text-muted-foreground border-border',
  },
  cancelled: {
    label: 'Cancelled',
    className: 'bg-muted text-muted-foreground border-border',
  },
};

export default function UsersSettings() {
  const { user, profile } = useAuth();
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [pendingInvites, setPendingInvites] = useState<Invitation[]>([]);
  const [loading, setLoading] = useState(true);
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<Role>('viewer');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Fetch team members and pending invites
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        // Fetch team members from profiles table
        const { data: profilesData, error: profilesError } = await supabase
          .from('profiles')
          .select('*')
          .order('created_at', { ascending: false });

        if (profilesError) throw profilesError;

        // Transform data to TeamMember format
        // WARNING: This role assignment happens CLIENT-SIDE ONLY
        // The backend MUST validate all role changes via Supabase RLS policies
        // DO NOT trust this value for authorization - it's UI-only
        const members: TeamMember[] = (profilesData || []).map((p, idx) => ({
          id: p.id,
          email: p.email || 'Unknown',
          display_name: p.display_name,
          role: (idx === 0 ? 'admin' : 'member') as Role,
          status: 'active' as Status,
          avatar_url: p.avatar_url,
        }));

        setTeamMembers(members);

        // Fetch pending invites
        const { data: invitesData, error: invitesError } = await supabase
          .from('invitations')
          .select('*')
          .order('created_at', { ascending: false });

        if (!invitesError && invitesData) {
          setPendingInvites(invitesData.map(inv => ({
            id: inv.id,
            email: inv.email,
            role: inv.role as Role,
            status: inv.status as Status,
            created_at: inv.created_at,
            expires_at: inv.expires_at,
          })));
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        // Fallback to current user if fetch fails
        if (user && profile) {
          setTeamMembers([{
            id: user.id,
            email: user.email || 'Unknown',
            display_name: profile.display_name,
            role: 'admin',
            status: 'active',
            avatar_url: profile.avatar_url,
          }]);
        }
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [user, profile]);

  const handleInvite = async () => {
    if (!inviteEmail) {
      toast.error('Please enter an email address');
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(inviteEmail)) {
      toast.error('Please enter a valid email address');
      return;
    }

    setIsSubmitting(true);

    try {
      // Call the edge function to send the invite
      const { data, error } = await supabase.functions.invoke('admin-invite', {
        body: {
          email: inviteEmail,
          role: inviteRole,
        },
      });

      if (error) {
        throw new Error(error.message || 'Failed to send invitation');
      }

      if (data?.error) {
        toast.error('Invitation failed', {
          description: data.error,
        });
        return;
      }

      toast.success('Invitation sent', {
        description: `An invitation has been sent to ${inviteEmail}`,
      });

      // Add to pending invites list
      setPendingInvites(prev => [{
        id: `pending-${Date.now()}`,
        email: inviteEmail,
        role: inviteRole,
        status: 'pending',
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      }, ...prev]);

      setInviteDialogOpen(false);
      setInviteEmail('');
      setInviteRole('viewer');
    } catch (err) {
      console.error('Invite error:', err);
      toast.error('Failed to send invitation', {
        description: err instanceof Error ? err.message : 'Please try again later',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRoleChange = async (memberId: string, newRole: Role) => {
    setIsSubmitting(true);

    try {
      // WARNING: This updates UI state only
      // Backend MUST validate role changes via RLS policies or edge functions
      // TODO: Call backend API to persist role change with proper authorization
      setTeamMembers(prev =>
        prev.map(m => m.id === memberId ? { ...m, role: newRole } : m)
      );

      toast.success('Role updated', {
        description: `User role has been changed to ${roleConfig[newRole].label}`,
      });

      setEditDialogOpen(false);
      setSelectedMember(null);
    } catch (err) {
      toast.error('Failed to update role');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRemoveMember = async (memberId: string) => {
    if (memberId === user?.id) {
      toast.error('You cannot remove yourself');
      return;
    }

    setIsSubmitting(true);

    try {
      setTeamMembers(prev => prev.filter(m => m.id !== memberId));

      toast.success('Member removed', {
        description: 'The user has been removed from the team',
      });

      setEditDialogOpen(false);
      setSelectedMember(null);
    } catch (err) {
      toast.error('Failed to remove member');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelInvite = async (inviteId: string) => {
    try {
      await supabase
        .from('invitations')
        .update({ status: 'cancelled' })
        .eq('id', inviteId);

      setPendingInvites(prev => prev.filter(inv => inv.id !== inviteId));
      toast.success('Invitation cancelled');
    } catch (err) {
      toast.error('Failed to cancel invitation');
    }
  };

  const openEditDialog = (member: TeamMember) => {
    setSelectedMember(member);
    setEditDialogOpen(true);
  };

  const activePendingInvites = pendingInvites.filter(inv => inv.status === 'pending');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-foreground">User Management</h1>
        <Button onClick={() => setInviteDialogOpen(true)} className="bg-primary hover:bg-primary/90">
          <UserPlus className="mr-2 h-4 w-4" />
          Invite User
        </Button>
      </div>

      {/* Role Legend */}
      <NILMPanel
        title="Role Permissions"
        icon={<Shield className="h-5 w-5" />}
        footer="Roles determine access levels for NILM data and system configuration"
      >
        <div className="grid gap-4 md:grid-cols-3 relative">
          <WaveformDecoration className="absolute top-0 right-0" />
          {Object.entries(roleConfig).map(([key, config]) => (
            <div key={key} className="space-y-1.5">
              <Badge variant="outline" className={config.className}>
                {config.label}
              </Badge>
              <p className="text-sm text-muted-foreground">
                {config.description}
              </p>
            </div>
          ))}
        </div>
      </NILMPanel>

      {/* Pending Invites */}
      {activePendingInvites.length > 0 && (
        <NILMPanel
          title="Pending Invites"
          icon={<Clock className="h-5 w-5" />}
          footer={`${activePendingInvites.length} pending invitation${activePendingInvites.length !== 1 ? 's' : ''}`}
        >
          <div className="space-y-2">
            {activePendingInvites.map((invite) => (
              <div
                key={invite.id}
                className="flex items-center justify-between py-3 px-4 rounded-lg bg-muted/30 border border-border"
              >
                <div className="flex items-center gap-3">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium text-foreground font-mono">{invite.email}</p>
                    <p className="text-xs text-muted-foreground">
                      Sent {formatDistanceToNow(new Date(invite.created_at), { addSuffix: true })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={roleConfig[invite.role].className}>
                    {roleConfig[invite.role].label}
                  </Badge>
                  <Badge variant="outline" className={statusConfig[invite.status].className}>
                    {statusConfig[invite.status].label}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCancelInvite(invite.id)}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </NILMPanel>
      )}

      {/* Team Members Table */}
      <NILMPanel
        title="Team Members"
        icon={<Users className="h-5 w-5" />}
        footer={`${teamMembers.length} team member${teamMembers.length !== 1 ? 's' : ''} with access to NILM monitoring`}
      >
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-muted/50">
                <TableHead className="text-muted-foreground">Name</TableHead>
                <TableHead className="text-muted-foreground">Email</TableHead>
                <TableHead className="text-muted-foreground">Role</TableHead>
                <TableHead className="text-muted-foreground">Status</TableHead>
                <TableHead className="text-right text-muted-foreground">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {teamMembers.map((member) => (
                <TableRow key={member.id} className="border-border hover:bg-muted/50">
                  <TableCell className="font-medium text-foreground">
                    {member.display_name || 'No name set'}
                    {member.id === user?.id && (
                      <Badge variant="outline" className="ml-2 text-xs">You</Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Mail className="h-4 w-4" />
                      <span className="font-mono text-sm">{member.email}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className={roleConfig[member.role].className}>
                      {roleConfig[member.role].label}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className={statusConfig[member.status].className}>
                      {statusConfig[member.status].label}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => openEditDialog(member)}
                      className="text-muted-foreground hover:text-foreground hover:bg-muted"
                      disabled={member.id === user?.id}
                    >
                      <Edit2 className="h-4 w-4 mr-1.5" />
                      Edit
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </NILMPanel>

      {/* Invite Dialog */}
      <Dialog open={inviteDialogOpen} onOpenChange={setInviteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite Team Member</DialogTitle>
            <DialogDescription>
              Send an invitation to join your Energy Monitor team. They will receive an email with a link to create their account.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="invite-email">Email Address</Label>
              <Input
                id="invite-email"
                type="email"
                placeholder="colleague@example.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="invite-role">Role</Label>
              <Select value={inviteRole} onValueChange={(v) => setInviteRole(v as Role)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="viewer">Viewer - Read-only access</SelectItem>
                  <SelectItem value="member">Member - View and analyze data</SelectItem>
                  <SelectItem value="admin">Admin - Full access</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleInvite} disabled={isSubmitting}>
              {isSubmitting && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              Send Invitation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Member Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Team Member</DialogTitle>
            <DialogDescription>
              Update role or remove {selectedMember?.display_name || selectedMember?.email}
            </DialogDescription>
          </DialogHeader>
          {selectedMember && (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Email</Label>
                <p className="text-sm text-muted-foreground font-mono">{selectedMember.email}</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="edit-role">Role</Label>
                <Select
                  value={selectedMember.role}
                  onValueChange={(v) => handleRoleChange(selectedMember.id, v as Role)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="viewer">Viewer</SelectItem>
                    <SelectItem value="member">Member</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}
          <DialogFooter className="flex justify-between">
            <Button
              variant="destructive"
              onClick={() => selectedMember && handleRemoveMember(selectedMember.id)}
              disabled={isSubmitting}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Remove
            </Button>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
