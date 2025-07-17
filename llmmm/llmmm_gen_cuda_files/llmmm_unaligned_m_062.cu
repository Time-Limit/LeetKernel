#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_62_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<62> mm_instantiator;

public:
  UnalignedM_62_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_62_MMInstantiatorWrapper__;

}  // namespace LLMMM
