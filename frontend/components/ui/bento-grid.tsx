import { cn } from "@/lib/utils";

export const BentoGrid = ({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "mx-auto grid max-w-7xl grid-cols-1 gap-1 md:auto-rows-[11rem] md:grid-cols-3",
        className,
      )}
    >
      {children}
    </div>
  );
};

export const BentoGridItem = ({
  className,
  title,
  description,
  header,
  icon,
}: {
  className?: string;
  title?: string | React.ReactNode;
  description?: string | React.ReactNode;
  header?: React.ReactNode;
  icon?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "row-span-1 flex flex-col justify-between rounded-lg border border-border bg-card p-2",
        className,
      )}
    >
      {header}
      <div>
        <div className="flex items-center gap-1.5">
          {icon}
          <div className="font-medium text-sm text-card-foreground">
            {title}
          </div>
        </div>
        {description && (
          <div className="text-xs text-muted-foreground line-clamp-1">
            {description}
          </div>
        )}
      </div>
    </div>
  );
};
